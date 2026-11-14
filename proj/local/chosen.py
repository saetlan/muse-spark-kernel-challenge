"""Lab winners accumulated for the scored builds (make_v7.build(CHOSEN, ...)); each entry names its job."""

PE9 = {  # job 4a1ab72284bc40ac (F12V at m = 1, F12M at m = 8/16)
    ('o', 8): {'kind': 'f12m', 'RTW': 2, 'RG': 1, 'KSW': 8, 'D': 1, 'F12': True, 'NT': 1},
    ('pooler', 1): {'kind': 'f12v', 'R': 4, 'RG': 4, 'KSW': 2, 'D': 1},
    ('pooler', 8): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 1},
    ('pooler', 16): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 2},
    ('qkv_gate', 1): {'kind': 'f12v', 'R': 3, 'RG': 3, 'KSW': 2, 'D': 2},
    ('qkv_gate', 8): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 4, 'F12': True, 'NT': 1},
    ('qkv_gate', 16): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 2},
    ('unpooler', 16): {'kind': 'f12m', 'RTW': 2, 'RG': 1, 'KSW': 8, 'D': 1, 'F12': True, 'NT': 2},
}
PE10 = {  # job 4922c5d5cb3545e9 (router wide Triton tiles)
    ('router', 32): {'kind': 'gemm', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 4},
    ('router', 128): {'kind': 'gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'mode': 1, 'warps': 8, 'stages': 3},
    ('router', 256): {'kind': 'gemm', 'BM': 256, 'BN': 64, 'BK': 64, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 3},
}
PE11 = {}  # job pending (WG at m = 128/256)

V6 = dict(PE9)
V7 = {**PE9, **PE10, **PE11}
