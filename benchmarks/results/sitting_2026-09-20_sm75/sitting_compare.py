"""Cross-leg comparison of a sitting: python sitting_compare.py <dir> <tagA> <tagB> [tagC ...]
Ratios are A/B (time_A / time_B): >1.00 means B is FASTER than A. Geomeans per config for
eight_config (rows are noise, per roadmap §10); scalar metrics for the interactive benches."""
import json, math, os, sys
d, tags = sys.argv[1], sys.argv[2:]
def load(tag, bench):
    p = os.path.join(d, f"{tag}_{bench}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
def gm(xs): xs = [x for x in xs if x and x > 0]; return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
def fmt(x): return "  n/a " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:6.3f}"
ref = tags[0]
print(f"legs: {tags}   (ratio = {ref}/leg; >1 = leg faster)\n")
# eight_config: per-config geomean of median ratios
E = {t: load(t, "eight_config") for t in tags}
if E[ref]:
    print("eight_config @512^2 — per-config geomean" + "".join(f" | {t:>9}" for t in tags[1:]) + "   [row min..max for last leg]")
    for cfg in E[ref]["results"]:
        line = f"  {cfg:14}"; rows = []
        for t in tags[1:]:
            if not E[t]: line += " |    n/a  "; continue
            rs = []
            for prog, a in E[ref]["results"][cfg].items():
                b = E[t]["results"].get(cfg, {}).get(prog)
                if a and b and a.get("median") and b.get("median") and a.get("status") == b.get("status"):
                    rs.append(a["median"] / b["median"])
            rows = rs; line += f" | {fmt(gm(rs))} n={len(rs):3}"
        if rows: line += f"   [{min(rows):.2f}..{max(rows):.2f}]"
        print(line)
    print()
def scalar_table(bench, getters):
    D = {t: load(t, bench) for t in tags}
    if not D[ref]: print(f"{bench}: no data for {ref}"); return
    print(f"{bench}" + "".join(f" | {t:>10}" for t in tags) + "  ratios vs " + ref)
    for name, g in getters:
        vals = []
        for t in tags:
            try: vals.append(float(g(D[t])) if D[t] else None)
            except Exception: vals.append(None)
        line = f"  {name:28}" + "".join(f" | {v:10.4f}" if v is not None else " |        n/a" for v in vals)
        line += "  " + " ".join(fmt(vals[0] / v) if (v and vals[0]) else " n/a " for v in vals[1:])
        print(line)
    print()
scalar_table("roi_scrub", [(k, (lambda k: lambda j: j["rows"]["cuda"][k])(k)) for k in ("whole", "roi_fixed", "roi_panning", "roi_pan_param")])
scalar_table("param_scrub", [(k, (lambda k: lambda j: j[k])(k)) for k in ("cold_ms", "scrub_median_ms", "static_median_ms", "recook_median_ms", "scrub_max_after_warmup_ms")])
scalar_table("region_recook", [(f"{r}/{k}", (lambda r, k: lambda j: j["rows"][r][k])(r, k)) for r in ("cuda/n50/2048", "cpu/n50/2048") for k in ("whole_all", "region_mid")])
scalar_table("cookqueue", [(k, (lambda k: lambda j: j[k])(k)) for k in ("solo_ms", "queued_idle_ms", "preempt_to_first_stmt_ms", "queued_under_load_ms")])
# counts: stable rows (api/frames/cuda groups) per scenario, diff of medians
C = {t: load(t, "counts_cuda") for t in tags}
if C[ref]:
    print("counts_cuda - stable rows that differ from", ref)
    base = C[ref]["runs"][0]["scenarios"]; moved = 0; frames_moved = 0
    for t in tags[1:]:
        if not C[t]: continue
        for sc, sd in C[t]["runs"][0]["scenarios"].items():
            for grp in ("api", "cuda", "frames"):
                for row, rv in sd.get(grp, {}).items():
                    bv = base.get(sc, {}).get(grp, {}).get(row)
                    if isinstance(rv, dict) and isinstance(bv, dict) and rv.get("stable") and bv.get("stable") and rv.get("median") != bv.get("median"):
                        if grp == "frames": frames_moved += 1; continue
                        print(f"  {t:10} {sc:12} {row:36} {bv.get('median')} -> {rv.get('median')}"); moved += 1
    print(f"  moved api/cuda rows: {moved}; moved frame rows (not listed): {frames_moved}")
