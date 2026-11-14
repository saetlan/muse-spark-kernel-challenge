"""Pre-submission lint, run on the Mac before any okc check or submit.

1. Triton global access (Triton 3.8 code_generator.py, global_lookup): a @triton.jit function may
   read a module global only if it is a module, another jit function, a Triton builtin or dtype,
   or a value created as `name = tl.constexpr(...)` (an annotation `name: tl.constexpr = ...` is
   NOT allowed). Allowed Python builtins: len list range float int isinstance getattr hasattr
   print min max. Globals inside argument defaults are fine.
2. tl.inline_asm_elementwise operands: every $N in a literal asm string must be below the number
   of constraints, and with pack=1 and 32-bit dtypes there must be one "=" output per dtype.
3. Constructs the judge may reject (TASK.md: patches timing, reads harness frames, touches
   harness I/O or secrets, edits loaded modules, imports the reference, starts processes or uses
   the network; rules: no encoded or compressed source).
4. CUDA graph breakers in every function reachable from kernel_fn: host syncs (.item, .cpu,
   .tolist, .numpy, synchronize), ops whose output size depends on values (nonzero, unique,
   bincount, masked_select, one-argument where), and caching keyed by id() or data_ptr().
5. CUDA sources (string literals with __global__): every <<<...>>> launch must pass a stream as
   its 4th argument, PyTorch's current stream, or graph capture breaks.

Run: .venv/bin/python moe/local/lint_submission.py moe/submission.py moe/lab_e1b.py
Exits 1 if anything is flagged.
"""

import ast
import re
import sys

TRITON_BUILTINS = {"len", "list", "range", "float", "int", "isinstance", "getattr", "hasattr",
                   "print", "min", "max"}
RISKY_IMPORTS = {"subprocess", "multiprocessing", "socket", "urllib", "requests", "http", "inspect",
                 "base64", "zlib", "lzma", "bz2", "gzip", "pickle", "marshal", "ctypes", "reference",
                 "checks", "signal", "threading"}
RISKY_CALLS = {"exec", "eval", "compile", "open", "__import__"}
RISKY_ATTRS = {("os", "system"), ("os", "popen"), ("os", "fork"), ("os", "execv"), ("os", "kill"),
               ("sys", "_getframe"), ("os", "putenv"), ("os", "unsetenv")}


def is_jit_decorator(node):
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr == "jit"
    return isinstance(target, ast.Name) and target.id == "jit"


def is_constexpr_call(node):
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Attribute) and f.attr == "constexpr") or (isinstance(f, ast.Name) and f.id == "constexpr")


def module_bindings(tree):
    """name -> 'module' | 'jit' | 'constexpr' | 'pyfunc' | 'class' | 'value' | 'from-import'."""
    kinds = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                kinds[(a.asname or a.name).split(".")[0]] = "module"
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                kinds[a.asname or a.name] = "from-import"
        elif isinstance(node, ast.FunctionDef):
            kinds[node.name] = "jit" if any(is_jit_decorator(d) for d in node.decorator_list) else "pyfunc"
        elif isinstance(node, ast.ClassDef):
            kinds[node.name] = "class"
        elif isinstance(node, ast.Assign):
            kind = "constexpr" if is_constexpr_call(node.value) else "value"
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        kinds[n.id] = kind
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            kinds[node.target.id] = "constexpr" if is_constexpr_call(node.value) else "annotated-value"
    return kinds


def local_names(fn):
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    for extra in (fn.args.vararg, fn.args.kwarg):
        if extra:
            names.add(extra.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
    return names


def default_value_nodes(fn):
    ids = set()
    for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]:
        ids.update(id(n) for n in ast.walk(d))
    return ids


def lint_triton_globals(path, tree, kinds, problems):
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if kinds.get(fn.name) != "jit":
            continue
        locals_ = local_names(fn)
        skip = default_value_nodes(fn)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)) or id(node) in skip:
                continue
            name = node.id
            if name in locals_ or name in TRITON_BUILTINS:
                continue
            kind = kinds.get(name)
            if kind in ("module", "jit", "constexpr"):
                continue
            why = {"value": "a plain global value; wrap it as name = tl.constexpr(...) or inline it",
                   "annotated-value": "annotated, not instantiated, as constexpr (unsupported)",
                   "pyfunc": "a non-jit Python function", "class": "a class",
                   "from-import": "imported by name; allowed only if it is a jit function or tl.constexpr",
                   None: "not defined at module level"}[kind]
            problems.append(f"{path}:{node.lineno}: jit function {fn.name} reads global {name!r}: {why}")


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def lint_inline_asm(path, tree, problems):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "inline_asm_elementwise"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        asm, cons = literal(kw.get("asm")) if "asm" in kw else None, literal(kw.get("constraints")) if "constraints" in kw else None
        if not isinstance(asm, str) or not isinstance(cons, str):
            problems.append(f"{path}:{node.lineno}: inline asm or constraints is not a literal string; can't check it")
            continue
        entries = [c.strip() for c in cons.split(",") if c.strip()]
        used = [int(m) for m in re.findall(r"\$(\d+)", asm)]
        if used and max(used) >= len(entries):
            problems.append(f"{path}:{node.lineno}: asm uses ${max(used)} but only {len(entries)} constraints")
        outputs = sum(1 for c in entries if c.startswith("="))
        dtype, pack = kw.get("dtype"), literal(kw.get("pack")) if "pack" in kw else None
        n_dtypes = len(dtype.elts) if isinstance(dtype, ast.Tuple) else 1
        dtype_names = [ast.unparse(e) for e in (dtype.elts if isinstance(dtype, ast.Tuple) else [dtype])] if dtype else []
        if pack == 1 and all(n.endswith(("int32", "uint32", "float32")) for n in dtype_names) and outputs != n_dtypes:
            problems.append(f"{path}:{node.lineno}: {outputs} '=' outputs for {n_dtypes} 32-bit dtype(s) at pack=1")


def lint_risky(path, tree, problems):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] in RISKY_IMPORTS:
                    problems.append(f"{path}:{node.lineno}: imports {m!r} (the judge may reject it)")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in RISKY_CALLS:
                problems.append(f"{path}:{node.lineno}: calls {f.id}() (the judge may reject it)")
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in RISKY_ATTRS:
                problems.append(f"{path}:{node.lineno}: calls {f.value.id}.{f.attr}()")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for t in targets:
                text = ast.unparse(t)
                if text.startswith("os.environ") or text.startswith("sys.modules"):
                    problems.append(f"{path}:{node.lineno}: modifies {text} (process-wide state)")
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in ("torch", "triton", "tl", "sglang"):
                    problems.append(f"{path}:{node.lineno}: assigns into module {text} (\"edits loaded modules\")")


SYNC_METHODS = {"item", "cpu", "tolist", "numpy", "synchronize", "nonzero", "data_ptr"}
VALUE_SIZED_TORCH = {"nonzero", "unique", "unique_consecutive", "bincount", "masked_select", "argwhere"}


def reachable_from(tree, root="kernel_fn"):
    """Module-level functions kernel_fn can reach through direct calls by name."""
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seen, todo = set(), [root] if root in funcs else []
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                todo.append(node.func.id)
    return [funcs[n] for n in sorted(seen)]


def lint_graph_safety(path, tree, problems):
    for fn in reachable_from(tree):
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            where = f"{path}:{node.lineno}: in {fn.name} (reachable from kernel_fn)"
            if isinstance(f, ast.Attribute) and f.attr in SYNC_METHODS:
                problems.append(f"{where}: .{f.attr}() syncs the host or keys on addresses; it breaks CUDA graph capture")
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "torch"
                    and f.attr in VALUE_SIZED_TORCH):
                problems.append(f"{where}: torch.{f.attr} has a value-dependent output size (host sync)")
            if (isinstance(f, ast.Attribute) and f.attr == "where" and isinstance(f.value, ast.Name)
                    and f.value.id == "torch" and len(node.args) == 1 and not node.keywords):
                problems.append(f"{where}: one-argument torch.where is nonzero (host sync)")
            if isinstance(f, ast.Name) and f.id == "id":
                problems.append(f"{where}: id() in the timed path suggests caching by object identity")


def lint_cuda_launches(path, tree, problems):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and "__global__" in node.value):
            continue
        for m in re.finditer(r"<<<(.*?)>>>", node.value, flags=re.S):
            args = [a for a in m.group(1).split(",") if a.strip()]
            if len(args) < 4:
                line = node.lineno + node.value[: m.start()].count("\n")
                problems.append(f"{path}:{line}: CUDA launch <<<{m.group(1).strip()}>>> passes no stream; "
                                "use at::cuda::getCurrentCUDAStream()")


def lint_file(path):
    src = open(path).read()
    problems = []
    if not src.startswith("# okc: problem="):
        problems.append(f"{path}:1: first line must be '# okc: problem=<name>'")
    tree = ast.parse(src, filename=path)
    kinds = module_bindings(tree)
    lint_triton_globals(path, tree, kinds, problems)
    lint_inline_asm(path, tree, problems)
    lint_risky(path, tree, problems)
    lint_graph_safety(path, tree, problems)
    lint_cuda_launches(path, tree, problems)
    if re.search(r"[A-Za-z0-9+/]{400,}={0,2}", src):
        problems.append(f"{path}: contains a long base64-like run (encoded source is not allowed)")
    return problems


if __name__ == "__main__":
    all_problems = []
    for p in sys.argv[1:]:
        found = lint_file(p)
        all_problems += found
        print(f"{p}: {'ok' if not found else f'{len(found)} problem(s)'}")
    for line in all_problems:
        print("  " + line)
    sys.exit(1 if all_problems else 0)
