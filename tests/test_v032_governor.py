"""v0.32 GOV-1 — memory/effort profiles on the CACHE-5 governor.

What has to be pinned, and why:

  * THE PRESET REACHES ALL THREE BUDGETS — the governor had three live, mutually-unaware
    budgets (`cache_budget_bytes`, `ResultCache._budget`, `governor_budget`). A "profile" that
    set only the arbitrated one would be a LABEL, not a policy: the frame cache would keep
    spilling against its constructor default no matter what the preset said.
  * ORDER-INDEPENDENCE — a host may pick a profile before or after it builds its caches. A
    budgets-are-constructor-arguments design silently only reaches caches created afterwards.
  * TIGHTENING TAKES EFFECT NOW — switching to `efficient` to get memory back must not wait
    for the next `put`, which on an idle session may never come.
  * BALANCED IS A TRUE NO-OP — the default must be byte-identical to not having profiles at
    all (invariant #7's posture), which is why its knobs are `None` rather than a restatement
    of numbers that would then drift from the real defaults.
  * NAMED AND REPORTABLE (S-5) — never silently auto-tune a box; `tex doctor` must be able to
    say which preset produced a number, or two users' reports are not comparable.
  * GOV-1 OWNS CACHE-7's THRESHOLD — the report names it as a profile knob, and a thriftier
    profile must checkpoint LESS often (each checkpoint is a whole frame held in RAM).
"""
from helpers import *

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_doctor, tex_memory, tex_results


def _fresh():
    tex_memory._reset_profile_for_test()


def test_v032_gov1_profile_table(r: SubTestResult):
    print("\n--- v0.32 GOV-1: the preset table ---")
    _fresh()
    names = tex_memory.profiles()
    if names == ("performance", "balanced", "efficient"):
        r.ok(f"GOV-1: three presets in increasing thrift {names}")
    else:
        r.fail("GOV-1 profiles", f"got {names}")
    if tex_memory.active_profile() == "balanced":
        r.ok("GOV-1: the default profile is `balanced` (today's behaviour)")
    else:
        r.fail("GOV-1 default", f"active is {tex_memory.active_profile()}")

    knobs = tex_memory.profile_knobs("balanced")
    if knobs["frame_mb"] is None and knobs["governor_frac"] is None:
        r.ok("GOV-1: `balanced` leaves the shipped budgets alone (a true no-op, not a restatement)")
    else:
        r.fail("GOV-1 balanced", f"balanced overrides budgets: {knobs}")

    # Thrift must be monotone across the axes a preset exists to trade.
    perf, eff = tex_memory.profile_knobs("performance"), tex_memory.profile_knobs("efficient")
    if perf["frame_mb"] > eff["frame_mb"] and perf["governor_frac"] > eff["governor_frac"]:
        r.ok(f"GOV-1: performance holds more than efficient "
             f"({perf['frame_mb']}MB/{perf['governor_frac']} vs "
             f"{eff['frame_mb']}MB/{eff['governor_frac']})")
    else:
        r.fail("GOV-1 monotonicity", f"performance={perf} efficient={eff}")
    if perf["checkpoint_ms"] < eff["checkpoint_ms"]:
        r.ok("GOV-1: a thriftier profile checkpoints LESS often (each one is a whole frame)")
    else:
        r.fail("GOV-1 checkpoint knob",
               f"performance={perf['checkpoint_ms']} efficient={eff['checkpoint_ms']}")

    try:
        tex_memory.set_profile("nonesuch")
        r.fail("GOV-1 validation", "an unknown profile name was accepted")
    except ValueError:
        r.ok("GOV-1: an unknown profile name raises ValueError")
    _fresh()


def test_v032_gov1_reaches_the_frame_cache_both_orders(r: SubTestResult):
    """ORDER-INDEPENDENCE + the double-budget gap. A preset must reach a cache armed BEFORE it
    was chosen and one armed AFTER, or it is only a label on the arbitrated budget."""
    print("\n--- v0.32 GOV-1: the preset reaches the frame cache, either order ---")
    _fresh()
    # (a) cache armed FIRST, profile chosen SECOND.
    before = tex_results.ResultCache(budget_mb=2048)
    tex_memory.register_result_cache(before, name="gov1-before")
    tex_memory.set_profile("efficient")
    want = tex_memory.profile_knobs("efficient")["frame_mb"] * (1 << 20)
    if before._budget == want:
        r.ok(f"GOV-1: a cache armed BEFORE the profile got its budget ({want >> 20} MB)")
    else:
        r.fail("GOV-1 apply order", f"armed-first cache kept {before._budget >> 20} MB")

    # (b) profile already active, cache armed SECOND.
    after = tex_results.ResultCache(budget_mb=2048)
    tex_memory.register_result_cache(after, name="gov1-after")
    if after._budget == want:
        r.ok(f"GOV-1: a cache armed AFTER the profile got its budget ({want >> 20} MB)")
    else:
        r.fail("GOV-1 apply order", f"armed-second cache kept {after._budget >> 20} MB")

    # (c) balanced restores the shipped default rather than pinning a number.
    tex_memory.set_profile("performance")
    perf = tex_memory.profile_knobs("performance")["frame_mb"] * (1 << 20)
    if before._budget == perf:
        r.ok("GOV-1: switching profiles re-applies to every armed cache")
    else:
        r.fail("GOV-1 switch", f"budget is {before._budget >> 20} MB, expected {perf >> 20}")
    _fresh()


def test_v032_gov1_tightening_evicts_now(r: SubTestResult):
    """A shrunk budget that waits for the next `put` leaves the cache over its stated cap for
    as long as the session is idle — which is exactly when a user switches to `efficient`."""
    print("\n--- v0.32 GOV-1: tightening takes effect immediately ---")
    _fresh()
    cache = tex_results.ResultCache(budget_mb=64)
    frame = torch.rand(1, 256, 256, 4)          # 1 MiB each
    for i in range(24):
        cache.put(f"f{i}", frame)
    before_bytes = cache.stats()["ram_bytes"]

    cache.set_budget(4)                          # 4 MiB — far below what is held
    st = cache.stats()
    if st["ram_bytes"] <= 4 * (1 << 20) or st["ram_entries"] <= 1:
        r.ok(f"GOV-1 set_budget: {before_bytes >> 20} MB -> {st['ram_bytes'] >> 20} MB "
             "without waiting for another put")
    else:
        r.fail("GOV-1 set_budget", f"still holding {st['ram_bytes'] >> 20} MB over a 4 MB cap")
    if st["spills"] > 0:
        r.ok(f"GOV-1 set_budget: the evicted frames spilled to disk ({st['spills']}), not dropped")
    else:
        r.fail("GOV-1 set_budget", "eviction dropped frames instead of spilling them")
    cache.clear(disk=True)
    _fresh()


def test_v032_gov1_owns_the_checkpoint_threshold(r: SubTestResult):
    """The report names the checkpoint threshold as a profile knob. CACHE-7 must read the
    ACTIVE profile at call time — a def-time default could never see a later `set_profile`."""
    print("\n--- v0.32 GOV-1: the profile owns CACHE-7's threshold ---")
    _fresh()
    src = torch.rand(1, 64, 64, 3)
    stages = [{"code": "@OUT = vec4(@IN.rgb * 1.05, 1.0);",
               "chain_input": (None if i == 0 else "IN"),
               "bindings": ({"IN": src} if i == 0 else {})} for i in range(6)]
    costs = {i: 80.0 for i in range(6)}          # each stage below 100 ms, above 50 ms

    placed = {}
    for name in tex_memory.profiles():
        tex_memory.set_profile(name)
        if CK.default_threshold_ms() != tex_memory.profile_knobs(name)["checkpoint_ms"]:
            r.fail("GOV-1 threshold", f"{name}: CACHE-7 did not read the profile")
            continue
        placed[name] = CK.plan_checkpoints(stages, costs=costs, px=64 * 64, settled=True)
        r.ok(f"GOV-1 {name}: threshold {CK.default_threshold_ms()} ms -> "
             f"{len(placed[name])} cut(s) {placed[name]}")

    # The property, stated as the trade rather than as absolute counts: a thriftier profile
    # holds FEWER checkpoints, because each one is a whole frame of RAM.
    if len(placed["performance"]) > len(placed["balanced"]) > len(placed["efficient"]):
        r.ok(f"GOV-1: checkpoint count is monotone in thrift "
             f"({len(placed['performance'])} > {len(placed['balanced'])} > "
             f"{len(placed['efficient'])})")
    else:
        r.fail("GOV-1 threshold effect",
               f"not monotone: { {k: len(v) for k, v in placed.items()} }")

    # An explicit argument still wins — a preset is a convenience, not a gag.
    tex_memory.set_profile("efficient")
    forced = CK.plan_checkpoints(stages, costs=costs, threshold_ms=10.0, px=64 * 64,
                                 settled=True)
    if forced:
        r.ok("GOV-1: an explicit threshold_ms still overrides the profile")
    else:
        r.fail("GOV-1 override", "an explicit threshold was ignored")
    _fresh()


def test_v032_gov1_is_named_and_reportable(r: SubTestResult):
    """S-5: never silently auto-tune a box. A profile must be nameable and reportable, or two
    users' numbers are not comparable — the same rule `arch_support.gate_profile` follows."""
    print("\n--- v0.32 GOV-1: named, and reportable through tex doctor ---")
    _fresh()
    tex_memory.set_profile("performance")
    facts = tex_doctor.collect_doctor_facts()
    mp = facts.get("memory_profile") or {}
    if mp.get("active") == "performance":
        r.ok("GOV-1: `tex doctor` reports the active profile")
    else:
        r.fail("GOV-1 doctor", f"memory_profile facts = {mp}")
    if mp.get("knobs", {}).get("checkpoint_ms"):
        r.ok(f"GOV-1: doctor carries the knobs too ({mp['knobs']})")
    else:
        r.fail("GOV-1 doctor", "the knobs are not reported")
    if set(mp.get("available") or ()) == set(tex_memory.profiles()):
        r.ok("GOV-1: doctor lists every available preset")
    else:
        r.fail("GOV-1 doctor", f"available = {mp.get('available')}")

    # Nothing selects a profile automatically — the default survives a governor call.
    _fresh()
    tex_memory.governor_budget("cpu")
    if tex_memory.active_profile() == "balanced":
        r.ok("GOV-1: nothing auto-selects a profile (S-5)")
    else:
        r.fail("GOV-1 S-5", f"a governor call changed the profile to "
               f"{tex_memory.active_profile()}")
    _fresh()


# ── audit fixes ─────────────────────────────────────────────────────────────────────────

def test_v032_gov1_balanced_restores_the_shipped_budget(r: SubTestResult):
    """BLOCKER. `frame_mb: None` meant "leave it alone", which is right on the FIRST apply and
    wrong on a switch BACK: efficient -> balanced left the 512 MB budget in place while
    `tex doctor` reported `balanced`. The report then describes a profile that is not being
    enforced, which is worse than having no report."""
    print("\n--- v0.32 GOV-1: balanced restores the shipped default ---")
    _fresh()
    cache = tex_results.ResultCache(budget_mb=1536)
    shipped = cache._budget
    tex_memory.register_result_cache(cache, name="gov1-roundtrip")
    if cache._budget == shipped:
        r.ok(f"GOV-1: arming under `balanced` leaves the shipped budget ({shipped >> 20} MB)")
    else:
        r.fail("GOV-1 round-trip", f"arming changed the budget to {cache._budget >> 20} MB")

    tex_memory.set_profile("efficient")
    tight = tex_memory.profile_knobs("efficient")["frame_mb"] * (1 << 20)
    if cache._budget == tight:
        r.ok(f"GOV-1: efficient applied ({tight >> 20} MB)")
    else:
        r.fail("GOV-1 round-trip", f"efficient left {cache._budget >> 20} MB")

    tex_memory.set_profile("balanced")
    if cache._budget == shipped:
        r.ok(f"GOV-1: balanced RESTORED the shipped budget ({shipped >> 20} MB), "
             "not just declined to change it")
    else:
        r.fail("GOV-1 round-trip",
               f"back on balanced the budget is {cache._budget >> 20} MB, "
               f"expected the shipped {shipped >> 20} MB")

    # The doctor must describe what is enforced.
    facts = tex_doctor.collect_doctor_facts().get("memory_profile") or {}
    if facts.get("active") == "balanced" and cache._budget == shipped:
        r.ok("GOV-1: `tex doctor` and the enforced budget agree after a round trip")
    else:
        r.fail("GOV-1 round-trip", f"doctor says {facts.get('active')}, "
               f"budget is {cache._budget >> 20} MB")
    _fresh()


def test_v032_gov1_governed_bytes_never_drifts(r: SubTestResult):
    """BLOCKER. `_bytes_by_dev` was maintained at three removal sites and one — the eviction
    loop — decremented `_ram_bytes` and forgot it. `governed_bytes(dev_type)` is what
    `arbitrate()` reads, so the per-device total ratcheted up on every eviction: measured at
    16000 MB for a cache actually holding 1984 (8.1x). The governor then evicted to the
    one-entry floor, which LOOKS like landing on budget and is really a 16x over-eviction that
    destroys the frame reuse CACHE-2/CACHE-9 exist to provide."""
    print("\n--- v0.32 GOV-1: governed_bytes never drifts from the truth ---")
    _fresh()
    cache = tex_results.ResultCache(budget_mb=4)          # tight: forces the eviction loop
    frame = torch.rand(1, 256, 256, 4)                    # 1 MiB
    for i in range(40):
        cache.put(f"k{i}", frame)

    def _truth():
        return sum(e[2] for e in cache._ram.values())

    got, want = cache.governed_bytes("cpu"), _truth()
    if got == want:
        r.ok(f"GOV-1: governed_bytes('cpu') == the real total after 40 puts + evictions "
             f"({got >> 20} MB)")
    else:
        r.fail("GOV-1 byte drift",
               f"governed_bytes says {got >> 20} MB, real total is {want >> 20} MB "
               f"({got / max(want, 1):.1f}x over)")

    cache.evict_bytes(2 << 20, dev_type="cpu")
    got, want = cache.governed_bytes("cpu"), _truth()
    if got == want:
        r.ok("GOV-1: still exact after an explicit evict_bytes")
    else:
        r.fail("GOV-1 byte drift", f"after evict: {got} vs {want}")

    if cache.governed_bytes(None) == cache.stats()["ram_bytes"] == want:
        r.ok("GOV-1: the device-split and whole-cache totals agree")
    else:
        r.fail("GOV-1 byte drift", "device-split and whole-cache totals disagree")

    cache.clear()
    if cache.governed_bytes("cpu") == 0 and cache.governed_bytes("cuda") == 0:
        r.ok("GOV-1: clear() zeroes the per-device totals too")
    else:
        r.fail("GOV-1 byte drift", "clear() left a non-zero per-device total")
    _fresh()


def test_v032_gov1_arbitrate_lands_on_budget_not_on_the_floor(r: SubTestResult):
    """The end-to-end consequence of the drift, which is the number PM-8 reports: with a truthful
    count the governor evicts to the BUDGET and leaves the cache useful; with the drift it
    evicts to the one-entry floor."""
    print("\n--- v0.32 GOV-1: arbitrate lands on budget, not on the floor ---")
    _fresh()
    cache = tex_results.ResultCache(budget_mb=4096)       # deliberately loose: the GOVERNOR evicts
    tex_memory.register_result_cache(cache, name="gov1-arb")
    frame = torch.rand(1, 256, 256, 4)                    # 1 MiB
    for i in range(64):
        cache.put(f"a{i}", frame)
    held = cache.stats()["ram_bytes"]
    budget = 16 << 20
    tex_memory.get_cache_registry().arbitrate("cpu", budget=budget)
    st = cache.stats()
    if st["ram_bytes"] <= budget and st["ram_entries"] > 1:
        r.ok(f"GOV-1: {held >> 20} MB -> {st['ram_bytes'] >> 20} MB in "
             f"{st['ram_entries']} entries, under a {budget >> 20} MB budget")
    elif st["ram_entries"] <= 1:
        r.fail("GOV-1 over-eviction",
               f"evicted to {st['ram_entries']} entry — the one-entry floor, not the budget")
    else:
        r.fail("GOV-1 arbitrate", f"{st['ram_bytes'] >> 20} MB still over {budget >> 20} MB")
    cache.clear(disk=True)
    _fresh()
