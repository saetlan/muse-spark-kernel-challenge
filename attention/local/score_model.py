"""Score model: per load (decode / verify x batch 1/16/32), SGLang total / ours total over 46 sliding + 16 full calls
(case weights), geomean over the 6 loads, from a check's per-shape medians. Reproduces the judge's speedup exactly
(L20: 1.4812). Run: python3 attention/local/score_model.py attention/local/<check>_result.json"""
import json, math, sys
def load(path):
    r = json.load(open(path))['result']
    t = {}
    for x in r['records']:
        if 'baseline_ns' in x:
            c = x['case']
            t[(c['load'], c['layer'])] = (x['median_ns'] / 1e3, x['baseline_ns'] / 1e3, c['weight'])
    return t, r['speedup']
def score(t, over=None):
    over = over or {}
    loads = sorted({k[0] for k in t})
    lg = 0
    for ld in loads:
        num = den = 0
        for layer in ('sliding', 'full'):
            ours, base, w = t[(ld, layer)]
            ours = over.get((ld, layer), ours)
            num += w * base; den += w * ours
        lg += math.log(num / den)
    return math.exp(lg / len(loads))
t, sp = load(sys.argv[1])
for k in sorted(t): print(k, '%.2f us ours / %.2f us SGLang' % t[k][:2])
print('judge speedup %.4f x, model %.4f x' % (sp, score(t)))
w1 = {('verify 16', 'full'): 178.3 * t[('verify 16','full')][0] / 228.8, ('verify 32', 'full'): 331.7 * t[('verify 32','full')][0] / 401.6,
      ('verify 16', 'sliding'): 24.5 * t[('verify 16','sliding')][0] / 28.8, ('verify 32', 'sliding'): 30.0 * t[('verify 32','sliding')][0] / 41.3,
      ('verify 1', 'full'): 29.6 * t[('verify 1','full')][0] / 32.7}
print('with W1 at L20 lab ratios: %.4f x' % score(t, w1))
print('--- scenarios on top of W1 ---')
def sc(name, extra):
    o = dict(w1); o.update(extra); print('%-55s %.3f x' % (name, score(t, o)))
sc('W1 only', {})
sc('+ full decode at 4.2 TB/s (Fd16 78, Fd32 156 us)', {('decode 16','full'): 78, ('decode 32','full'): 156})
sc('+ Fd1 at 10 us', {('decode 1','full'): 10})
sc('+ all sliding at 6 us (verify 32 at 9 us)', {(l,'sliding'): (9 if l=='verify 32' else 6) for l in ['decode 1','decode 16','decode 32','verify 1','verify 16','verify 32']})
sc('+ Fv16 100, Fv32 190 us', {('verify 16','full'): 100, ('verify 32','full'): 190})
allx = {('decode 16','full'): 78, ('decode 32','full'): 156, ('decode 1','full'): 10, ('verify 16','full'): 100, ('verify 32','full'): 190, ('verify 1','full'): 14}
allx.update({(l,'sliding'): (9 if l=='verify 32' else 6) for l in ['decode 1','decode 16','decode 32','verify 1','verify 16','verify 32']})
sc('all of the above (+ Fv1 14 us)', allx)
