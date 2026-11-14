"""RG configs: per router m (32/128/256) the (RTC, NTW, CS, XS, D) instances; NTW follows from m and RTC
(MT = (4 / RTC) * 8 * NTW = m). Grid: (16 / RTC) * CS CTAs (64-256)."""

MS = (32, 128, 256)


def ntw_for(m, rtc):
    return m // (8 * (4 // rtc))


def candidates(m):
    out = []
    for rtc in (2, 1):
        ntw = ntw_for(m, rtc)
        if ntw != 1 and ntw % 2:
            continue
        for cs in (8, 16):
            for d in (2, 1):
                out.append(dict(RTC=rtc, NTW=ntw, CS=cs, XS=3, D=d))
    return out


def instance(cfg):
    return (cfg["RTC"], cfg["NTW"], cfg["CS"], cfg["XS"], cfg["D"])


LAB = tuple(dict.fromkeys(instance(c) for m in MS for c in candidates(m)))


PROFILE = ((1, 4, 16, 3, 2),)   # PE14 best at router m128 (Modal profiling)
