


# ---------------------------------------------------------------------------------------------
# Lab L57 (check runs only), on v15 (the timed path = v15 byte for byte): UD5 anatomy (make_anatud5.py,
# lab_anat_ud5_block.py) on Sd1 / Sd16 / Sd32 at v15's routes: where sliding decode still spends its time after AF70.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5757)
    _log(f"L57: timed path v15: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    _anat_ud5(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
