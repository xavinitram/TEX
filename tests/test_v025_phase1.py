"""
v0.25 Phase 1 — "Remember frames" (results become first-class).

ENG-12  buffer ownership & immutability contract (this file's first block): a cook output
        is born FROZEN (inference-flagged → torch raises on any in-place write); a frame
        cache stores it frozen or version-stamps a normal frame and re-verifies at re-entry.
        M-5 out= reuse can never target a binding/cached frame; a scatter into a re-entered
        frame COW-clones. Lands FIRST — before any frame is cached.
CACHE-1  lineage keys: H(program_fp × params × upstream × frame × device × precision ×
        env_epoch × flags). Device+precision mandatory; a cross-device/precision/env cook
        mints a DISTINCT key (a stale-envelope hit is never served — invariant #9).
CACHE-2  tex_results.ResultCache — the engine frame cache (RAM byte-budget + disk spill,
        keyed by CACHE-1, frames frozen per ENG-12). Differential: a cached frame == a
        freshly cooked one, bit-exact; spill/restore round-trips bit-exact; eviction bounded.
CACHE-3  warm_state.json (graph-capture verdicts/blacklists + backend probes) + prewarm.
CACHE-4  layered cache epochs (AST ⊑ CODEGEN ⊑ VERDICT) so a codegen-only edit no longer
        cold-starts the parsed-program (.pkl) tier; mono-hash demoted to a completeness
        tripwire; a fail-safe oracle spot-check catches a wrong no-bump.

CPU-pinned for determinism; CUDA looped when present.
"""
from helpers import *  # noqa: F401,F403  (SubTestResult, torch, compile_and_run, make_img)
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
from TEX_Wrangle import tex_engine

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def _raises_inplace(t) -> bool:
    """True if an in-place op on `t` raises. NOTE: on torch 2.12 the raise does NOT roll back the
    write (the op lands, then raises), so this only proves torch's loud tripwire fires — NOT that
    the value is preserved. CACHE-2's real consumer guarantee is copy-on-read, tested separately."""
    try:
        t.add_(1.0)
        return False
    except Exception:
        return True


# ── ENG-12: buffer ownership & immutability contract ──────────────────────────

def test_eng12_output_is_born_frozen(r: SubTestResult):
    print("\n--- ENG-12: a cook output is born frozen (the floor) ---")
    # disown=False: no de-alias clone, so the raw tier output is observed directly.
    for dev in _DEVICES:
        try:
            res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.2, 1.0);",
                                  {"A": torch.rand(1, 4, 4, 4)},
                                  device_mode=dev, disown=False)
            out = res.outputs["OUT"]
            assert tex_engine.is_frozen(out), "cook output is not inference-flagged"
            assert _raises_inplace(out), "in-place write to a cook output did not raise"
            r.ok(f"[{dev}] cook output is frozen + torch raises on in-place write")
        except Exception as e:
            r.fail(f"[{dev}] cook output born frozen", f"{type(e).__name__}: {e}")


def test_eng12_two_strata(r: SubTestResult):
    print("\n--- ENG-12: frozen vs normal frame version stamps ---")
    # Stratum 1: a frozen frame — immutable, stamp is a constant 0, verify always True.
    try:
        frozen = tex_engine.frozen_copy(torch.rand(2, 3))
        assert tex_engine.is_frozen(frozen)
        assert tex_engine.frame_version(frozen) == 0
        assert tex_engine.verify_unmutated(frozen, 0)
        assert _raises_inplace(frozen), "frozen_copy is writable"
        r.ok("stratum 1: frozen frame is immutable, stamp==0, verify holds")
    except Exception as e:
        r.fail("ENG-12 stratum 1 (frozen)", f"{type(e).__name__}: {e}")

    # frozen_copy hard-freezes a NORMAL source without freezing the source itself.
    try:
        src = torch.rand(2, 3)
        fc = tex_engine.frozen_copy(src)
        assert tex_engine.is_frozen(fc) and not _raises_inplace(src), \
            "frozen_copy froze its source or produced a mutable copy"
        r.ok("frozen_copy hard-freezes a normal tensor; source stays mutable")
    except Exception as e:
        r.fail("ENG-12 frozen_copy", f"{type(e).__name__}: {e}")

    # Stratum 2: a normal frame carries torch's mutation counter; verify DETECTS a write.
    try:
        normal = torch.rand(2, 3)
        stamp = tex_engine.frame_version(normal)
        assert tex_engine.verify_unmutated(normal, stamp), "unmutated normal frame failed verify"
        normal.mul_(2.0)  # an in-place write bumps _version
        assert not tex_engine.verify_unmutated(normal, stamp), \
            "verify did not detect an in-place mutation"
        r.ok("stratum 2: normal frame version-stamp-and-verify detects a mutation")
    except Exception as e:
        r.fail("ENG-12 stratum 2 (normal)", f"{type(e).__name__}: {e}")

    # freeze() is idempotent: a frozen input is returned unchanged; a normal one is frozen.
    try:
        frozen = tex_engine.frozen_copy(torch.rand(2, 3))
        assert tex_engine.freeze(frozen) is frozen, "freeze re-copied an already-frozen frame"
        assert tex_engine.is_frozen(tex_engine.freeze(torch.rand(2, 3))), "freeze(normal) not frozen"
        r.ok("freeze() idempotent: frozen passthrough, normal hard-frozen")
    except Exception as e:
        r.fail("ENG-12 freeze idempotent", f"{type(e).__name__}: {e}")


def test_eng12_frozen_frame_reenters_scatter(r: SubTestResult):
    print("\n--- ENG-12: a frozen (cached) frame survives re-entry into a scattering cook ---")
    # A cached frame re-enters a later cook AS an input binding. A scatter into that name
    # must COW-clone before its first write, so the frozen input is neither written (which
    # would raise) nor observably mutated. This is what makes a frozen frame safe to cache
    # and hand back — the one residual write into a binding is already ownership-guarded.
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    code = "@OUT = @A;\n@OUT[0.0, 0.0] = vec3(9.0);"
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    for dev in _DEVICES:
        try:
            # Produce a frozen frame the way a cache would hold one.
            frame = tex_engine.frozen_copy(torch.rand(1, 4, 4, 3,
                                           device=("cuda" if dev == "cuda" else "cpu")))
            snapshot = frame.clone()  # a normal copy to compare against afterwards
            tm = TypeChecker(binding_types={}, source=code).check(prog)
            out = Interpreter().execute(prog, {"A": frame}, tm, device=dev,
                                        output_names=["OUT"], source=code)
            assert torch.equal(frame, snapshot), "the frozen (cached) frame was mutated by a scatter"
            # and the write did land on the owned copy, not the input
            assert abs(out["OUT"][0, 0, 0, 0].item() - 9.0) < 1e-6, "scatter did not write the output"
            r.ok(f"[{dev}] frozen frame unmutated after re-entering a scattering cook")
        except Exception as e:
            r.fail(f"[{dev}] ENG-12 frozen re-entry", f"{type(e).__name__}: {e}")


# ── CACHE-1: lineage keys ─────────────────────────────────────────────────────

def test_cache1_key_construction(r: SubTestResult):
    print("\n--- CACHE-1: lineage key construction + mandatory components ---")
    from TEX_Wrangle import tex_results as R
    base = dict(program_fp="abc123", device="cpu", precision="fp32")

    # Determinism: identical ingredients -> identical key.
    try:
        assert R.lineage_key(**base) == R.lineage_key(**base)
        r.ok("deterministic: same ingredients -> same key")
    except Exception as e:
        r.fail("CACHE-1 determinism", f"{type(e).__name__}: {e}")

    # Device and precision are MANDATORY (None is an error, never a wildcard).
    try:
        for bad in (dict(base, device=None), dict(base, precision=None)):
            raised = False
            try:
                R.lineage_key(**bad)
            except ValueError:
                raised = True
            assert raised, "a None device/precision did not raise"
        # program_fp None also raises
        raised = False
        try:
            R.lineage_key(program_fp=None, device="cpu", precision="fp32")
        except ValueError:
            raised = True
        assert raised
        r.ok("device / precision / program_fp are mandatory (None raises)")
    except Exception as e:
        r.fail("CACHE-1 mandatory components", f"{type(e).__name__}: {e}")

    # Every keyed component DISCRIMINATES: changing any one changes the key.
    try:
        k0 = R.lineage_key(**base)
        variants = {
            "device": dict(base, device="cuda:0"),
            "precision": dict(base, precision="fp16"),
            "program_fp": dict(base, program_fp="def456"),
            "params": dict(base, params={"strength": 0.5}),
            "frame": dict(base, frame=7),
            "canvas": dict(base, canvas=(512, 512)),
            "upstream": dict(base, upstream=("upkey1",)),
            "flags": dict(base, flags=("out:OUT",)),
            "quality": dict(base, quality="preview"),
        }
        for label, kw in variants.items():
            assert R.lineage_key(**kw) != k0, f"changing {label} did not change the key"
        # and two DIFFERENT param values differ from each other
        assert (R.lineage_key(**dict(base, params={"s": 0.5}))
                != R.lineage_key(**dict(base, params={"s": 0.6})))
        r.ok(f"all {len(variants)} components discriminate + param value discriminates")
    except Exception as e:
        r.fail("CACHE-1 discrimination", f"{type(e).__name__}: {e}")

    # env_epoch carries torch identity and is folded into the key.
    try:
        ep = R.env_epoch()
        import torch
        assert torch.__version__.split("+")[0] in ep, "env_epoch missing torch version"
        r.ok(f"env_epoch folds torch/GPU/code identity ({ep[:40]}…)")
    except Exception as e:
        r.fail("CACHE-1 env_epoch", f"{type(e).__name__}: {e}")


def test_cache1_not_a_content_hash(r: SubTestResult):
    print("\n--- CACHE-1: a lineage key is NOT a pixel content hash ---")
    # The crux of CACHE-1: a tensor input enters by its UPSTREAM key, never its pixels. So
    # two cooks of the same program+params+device with DIFFERENT input pixels and the same
    # (empty) upstream produce the SAME lineage key — a lineage key identifies the cook that
    # produced a frame, it does not re-sample the frame (that is the collision-prone
    # sampling hash CACHE-1 stops trusting for reuse). Different upstream keys separate them.
    for dev in _DEVICES:
        try:
            code = "f$s = 1.2;\n@OUT = vec4(@A.rgb * $s, 1.0);"
            r1 = tex_engine.cook(code, {"A": torch.rand(1, 8, 8, 4)}, device_mode=dev,
                                 want_lineage=True)
            r2 = tex_engine.cook(code, {"A": torch.rand(1, 8, 8, 4)}, device_mode=dev,
                                 want_lineage=True)  # DIFFERENT pixels, same everything else
            assert r1.lineage is not None and r2.lineage is not None
            assert r1.lineage["OUT"] == r2.lineage["OUT"], \
                "lineage key depended on input pixel content"
            r.ok(f"[{dev}] identical program+params -> identical key despite different pixels")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-1 not-a-content-hash", f"{type(e).__name__}: {e}")


def test_cache1_engine_integration(r: SubTestResult):
    print("\n--- CACHE-1: engine attaches per-output lineage only when asked ---")
    code2 = "f$s = 1.2;\n@OUT = vec4(@A.rgb * $s, 1.0);\n@ALT = @A * 0.5;"
    # Default path: lineage is None (invariant #7 — the ComfyUI cook pays nothing).
    try:
        res = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 4, 4, 4)},
                              device_mode="cpu")
        assert res.lineage is None, "lineage attached on the default path"
        r.ok("default cook (want_lineage=False) -> lineage is None")
    except Exception as e:
        r.fail("CACHE-1 default neutral", f"{type(e).__name__}: {e}")

    # want_lineage=True: a key per output, distinct across outputs, stable across cooks.
    try:
        a = tex_engine.cook(code2, {"A": torch.rand(1, 4, 4, 4)}, device_mode="cpu",
                            want_lineage=True)
        b = tex_engine.cook(code2, {"A": torch.rand(1, 4, 4, 4)}, device_mode="cpu",
                            want_lineage=True)
        assert set(a.lineage) == {"OUT", "ALT"}, f"outputs {set(a.lineage)}"
        assert a.lineage["OUT"] != a.lineage["ALT"], "two outputs share a key"
        assert a.lineage == b.lineage, "same program+params not stable across cooks"
        r.ok("per-output keys: distinct across outputs, stable across cooks")
    except Exception as e:
        r.fail("CACHE-1 per-output keys", f"{type(e).__name__}: {e}")

    # A different $param value re-keys every output.
    try:
        c = tex_engine.cook(code2, {"A": torch.rand(1, 4, 4, 4), "s": 0.7}, device_mode="cpu",
                            want_lineage=True)
        base = tex_engine.cook(code2, {"A": torch.rand(1, 4, 4, 4), "s": 1.2},
                               device_mode="cpu", want_lineage=True)
        assert c.lineage["OUT"] != base.lineage["OUT"], "a param change did not re-key"
        r.ok("a changed $param re-keys the output")
    except Exception as e:
        r.fail("CACHE-1 param re-key", f"{type(e).__name__}: {e}")

    # Cross-device: a cuda cook and a cpu cook of the same program must NOT share a key
    # (invariant #9 — a cross-device hit is never served).
    if _CUDA:
        try:
            cpu = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 4, 4, 4)},
                                  device_mode="cpu", want_lineage=True)
            gpu = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 4, 4, 4)},
                                  device_mode="cuda", want_lineage=True)
            assert cpu.lineage["OUT"] != gpu.lineage["OUT"], "cpu and cuda cooks shared a key"
            r.ok("cross-device cooks mint distinct keys (no cross-device hit)")
        except Exception as e:
            r.fail("CACHE-1 cross-device", f"{type(e).__name__}: {e}")


# ── CACHE-2: the frame cache (ResultCache) ────────────────────────────────────

import os as _os


def _fresh_cache(**kw):
    """A ResultCache pointed at an isolated scratch dir so tests never touch the shared
    results/ tier (and never see each other's spilled frames)."""
    from TEX_Wrangle import tex_results as R
    scratch = _os.path.join(_os.environ.get("TEX_CACHE_DIR", "."), "results_test")
    c = R.ResultCache(cache_dir=scratch, **kw)
    c.clear(disk=True)
    return c


def test_cache2_hit_is_bit_exact(r: SubTestResult):
    print("\n--- CACHE-2: a served frame == a freshly cooked one (bit-exact) + copy-on-read safe ---")
    for dev in _DEVICES:
        try:
            c = _fresh_cache()
            res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.3 + 0.1, 1.0);",
                                  {"A": torch.rand(1, 16, 16, 4)}, device_mode=dev,
                                  want_lineage=True)
            key = res.lineage["OUT"]
            frame = res.outputs["OUT"]
            c.put(key, frame, canvas=(16, 16))
            got = c.get(key)
            assert got is not None and torch.equal(got, frame), "hit not bit-exact"
            assert c.get("nonexistent-key") is None, "miss did not return None"
            # COPY-ON-READ is the real guarantee: the default get() returns an OWNED, mutable copy —
            # NOT the stored buffer — so a consumer's in-place write cannot corrupt the cache. (A
            # frozen frame is NOT write-proof on torch 2.12: an in-place op lands the write, then
            # raises. verify_unmutated can't catch that on a frozen master — copy-on-read is what does.)
            assert got is not c._ram[key][0], "default get() returned the stored buffer (not a copy)"
            snapshot = frame.clone()
            try:
                got.mul_(0.0)            # a consumer mutates the served frame (would corrupt a shared buffer)
            except Exception:
                pass
            again = c.get(key)
            assert torch.equal(again, snapshot), "the cache was corrupted by a write to a served frame"
            # copy=False is the opt-in zero-copy path: it hands back the frozen master.
            master = c.get(key, copy=False)
            assert master is c._ram[key][0] and tex_engine.is_frozen(master), \
                "copy=False did not return the frozen master"
            r.ok(f"[{dev}] hit bit-exact; copy-on-read protects the cache; copy=False = the master")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-2 hit bit-exact", f"{type(e).__name__}: {e}")


def test_cache2_spill_restore_bit_exact(r: SubTestResult):
    print("\n--- CACHE-2: RAM budget → spill → restore is bit-exact + bounded ---")
    for dev in _DEVICES:
        try:
            # budget holds ~two 64x64x4 fp32 frames (64KB each) but not four → forces spills.
            c = _fresh_cache(budget_mb=0.13)
            frames, keys = [], []
            for i in range(4):
                res = tex_engine.cook("f$s = 1.0;\n@OUT = vec4(@A.rgb * $s, 1.0);",
                                      {"A": torch.rand(1, 64, 64, 4), "s": 1.0 + i},
                                      device_mode=dev, want_lineage=True)
                keys.append(res.lineage["OUT"])
                frames.append(res.outputs["OUT"].clone())  # a normal snapshot to compare
                c.put(keys[-1], res.outputs["OUT"], canvas=(64, 64))
            st = c.stats()
            # Under budget, except the load-bearing carve-out: a single entry may exceed it
            # (a frame bigger than the whole budget still serves once rather than never).
            assert st["ram_bytes"] <= c._budget or st["ram_entries"] <= 1, \
                f"RAM over budget with >1 entry: {st['ram_bytes']} > {c._budget}"
            assert st["spills"] >= 1, "no frame was spilled under a tight budget"
            # the earliest frames were spilled to disk — restore one and check bit-exactness
            restored = c.get(keys[0])
            assert restored is not None, "spilled frame could not be restored"
            assert restored.device.type == ("cuda" if dev == "cuda" else "cpu"), \
                "restored to the wrong device"
            assert torch.equal(restored.cpu(), frames[0].cpu()), "spill/restore was not bit-exact"
            assert st["restores"] >= 0
            r.ok(f"[{dev}] spill/restore bit-exact; RAM stayed under budget ({st['spills']} spills)")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-2 spill/restore", f"{type(e).__name__}: {e}")


def test_cache2_verify_drop_and_replace(r: SubTestResult):
    print("\n--- CACHE-2: mutation-drop (ENG-12 stratum 2) + replace accounting ---")
    from TEX_Wrangle import tex_results as R
    # White-box: a NORMAL (unfrozen) frame injected into the RAM tier is version-stamped; an
    # in-place mutation must make get() drop it rather than serve a corrupted frame. (This is the
    # stratum-2 defense for a host that stores mutable frames; it is inert for the frozen masters
    # put() produces — those are protected by copy-on-read.) copy=False inspects the stored master.
    try:
        c = _fresh_cache()
        normal = torch.rand(1, 8, 8, 4)
        stamp = tex_engine.frame_version(normal)
        nbytes = normal.untyped_storage().nbytes()
        c._ram["k"] = [normal, stamp, nbytes, str(normal.device), None]
        c._ram_bytes += nbytes
        assert c.get("k", copy=False) is normal, "an unmutated normal frame was not served"
        normal.mul_(2.0)  # tamper
        assert c.get("k") is None, "a mutated frame was served instead of dropped"
        r.ok("a version-stamped normal frame is dropped once mutated (never served corrupt)")
    except Exception as e:
        r.fail("CACHE-2 verify-drop", f"{type(e).__name__}: {e}")

    # Replace: re-putting a key updates the frame and keeps byte accounting exact.
    try:
        c = _fresh_cache()
        f1 = tex_engine.frozen_copy(torch.rand(1, 8, 8, 4))
        f2 = tex_engine.frozen_copy(torch.rand(1, 8, 8, 4))
        c.put("k", f1)
        b1 = c._ram_bytes
        c.put("k", f2)
        assert len(c._ram) == 1 and c._ram_bytes == b1, "replace mis-accounted bytes"
        assert torch.equal(c.get("k"), f2), "replace did not update the frame"
        r.ok("re-put replaces the frame with exact byte accounting")
    except Exception as e:
        r.fail("CACHE-2 replace", f"{type(e).__name__}: {e}")


# ── CACHE-3: warm-tier persistence + prewarm ──────────────────────────────────

def test_cache3_warm_state_roundtrip(r: SubTestResult):
    print("\n--- CACHE-3: warm_state.json round-trips verdicts across a 'restart' ---")
    from TEX_Wrangle.tex_runtime import warm_state, graphed, compiled
    try:
        # Seed a capturability verdict (persists) plus a backend probe + compile-blacklist entry
        # (both deliberately NOT persisted — inert/transient). Only capturability round-trips.
        graphed._capturable_memo["fp_warm_test"] = (True, 7)
        graphed._capturable_memo["fp_nocap_test"] = (False, 0)
        compiled._backend_status[("inductor", "warmtest")] = True
        compiled._compile_blacklist["fp_bad_warm_test"] = None
        warm_state.persist(force=True)

        # Simulate a restart: forget the load latch and wipe the live tables.
        warm_state._reset_for_test()
        graphed._capturable_memo.pop("fp_warm_test", None)
        graphed._capturable_memo.pop("fp_nocap_test", None)
        compiled._backend_status.pop(("inductor", "warmtest"), None)
        compiled._compile_blacklist.pop("fp_bad_warm_test", None)

        warm_state.load()
        assert graphed._capturable_memo.get("fp_warm_test") == (True, 7), \
            "capturable=True verdict not restored"
        assert graphed._capturable_memo.get("fp_nocap_test") == (False, 0), \
            "capturable=False verdict not restored"
        # Backend probes + compile blacklist are inert/transient -> not persisted.
        assert ("inductor", "warmtest") not in compiled._backend_status, \
            "an (inert) backend probe was persisted"
        assert "fp_bad_warm_test" not in compiled._compile_blacklist, \
            "the (transient) compile blacklist was persisted"
        r.ok("capturability verdicts persist; inert backend + transient blacklist do NOT")
    except Exception as e:
        r.fail("CACHE-3 warm_state round-trip", f"{type(e).__name__}: {e}")
    finally:
        graphed._capturable_memo.pop("fp_warm_test", None)
        graphed._capturable_memo.pop("fp_nocap_test", None)
        compiled._backend_status.pop(("inductor", "warmtest"), None)
        compiled._compile_blacklist.pop("fp_bad_warm_test", None)


def test_cache3_version_tag_guard(r: SubTestResult):
    print("\n--- CACHE-3: a warm_state from another environment is ignored ---")
    from TEX_Wrangle.tex_runtime import warm_state, graphed
    import json as _json
    try:
        p = warm_state._path()
        assert p is not None
        with open(p, "w", encoding="utf-8") as f:
            _json.dump({"version": "some-other-gpu_9.9.9",
                        "capturable": {"fp_alien": [True, 3]},
                        "backend": {}, "compile_blacklist": []}, f)
        warm_state._reset_for_test()
        graphed._capturable_memo.pop("fp_alien", None)
        warm_state.load()
        assert "fp_alien" not in graphed._capturable_memo, \
            "a foreign-GPU warm_state verdict was wrongly adopted"
        r.ok("a warm_state tagged for another GPU/torch is ignored (arch-specific)")
    except Exception as e:
        r.fail("CACHE-3 version-tag guard", f"{type(e).__name__}: {e}")
    finally:
        graphed._capturable_memo.pop("fp_alien", None)


def test_cache3_prewarm(r: SubTestResult):
    print("\n--- CACHE-3: prewarm materializes codegen + seeds capturability ---")
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_compiler.types import TEXType
    from TEX_Wrangle.tex_runtime import warm_state
    import os as _os
    try:
        warm_state._reset_for_test()
        progs = [
            ("@OUT = vec4(@A.rgb * 1.5, 1.0);", {"A": TEXType.VEC4}),
            ("@OUT = @A * 0.5 + 0.25;", {"A": TEXType.VEC4}),
        ]
        summary = tex_api.prewarm(progs, shapes=[(1, 256, 256)], device="cpu",
                                  compile_mode="none")
        assert summary["programs"] == 2, f"prewarmed {summary['programs']} programs"
        assert summary["codegen"] >= 1, "no codegen fn was materialized"
        # warm_state.json should exist after a prewarm (persist force=True).
        p = warm_state._path()
        assert p is not None and _os.path.exists(p), "prewarm did not write warm_state.json"
        r.ok(f"prewarm: {summary['programs']} programs, {summary['codegen']} codegen warmed")
    except Exception as e:
        r.fail("CACHE-3 prewarm", f"{type(e).__name__}: {e}")


# ── CACHE-4: layered cache epochs ─────────────────────────────────────────────

# The complete watched compiler/runtime set (AST ∪ CODEGEN). The tripwire asserts the epoch
# partitions cover exactly this — so adding a compiler file without assigning it to an epoch
# reds the suite instead of letting it fall out of every cache key.
_WATCHED = {
    "ast_nodes.py", "lexer.py", "parser.py", "type_checker.py", "optimizer.py",
    "stdlib_signatures.py", "interpreter.py", "codegen.py", "codegen_stdfns.py",
    "stdlib.py", "noise.py", "tex_fusion.py",
}


def test_cache4_epoch_tripwire(r: SubTestResult):
    print("\n--- CACHE-4: layered epochs — completeness tripwire + partition shape ---")
    from TEX_Wrangle import tex_cache as C
    try:
        parts = C.epoch_partitions()
        ast = {p.name for p in parts["ast"]}
        cg = {p.name for p in parts["codegen"]}
        # AST and CODEGEN must be disjoint (nesting carries the AST→codegen dependency, not
        # membership); their union must be exactly the watched set (nothing added un-assigned).
        assert ast.isdisjoint(cg), f"file in both AST and CODEGEN partitions: {ast & cg}"
        assert ast | cg == _WATCHED, \
            f"epoch partitions don't cover the watched set (± {(ast | cg) ^ _WATCHED})"
        # every listed epoch file exists on disk (a typo'd path would silently hash its name)
        for group in parts.values():
            for p in group:
                assert p.exists(), f"epoch file missing on disk: {p}"
        r.ok("AST/CODEGEN disjoint, union == watched set, every epoch file exists")
    except Exception as e:
        r.fail("CACHE-4 tripwire", f"{type(e).__name__}: {e}")


def test_cache4_layering(r: SubTestResult):
    print("\n--- CACHE-4: the three epochs are distinct + nested ---")
    from TEX_Wrangle import tex_cache as C
    try:
        a, c, v = C.ast_epoch(), C.codegen_epoch(), C.verdict_epoch()
        assert len({a, c, v}) == 3, "epochs are not distinct"
        assert all(isinstance(e, str) and len(e) == 16 for e in (a, c, v)), "malformed epoch"
        # Nesting is real: the codegen epoch folds the AST epoch, so a hypothetical AST change
        # flows into it (this is what keeps a .cg from surviving an AST-file edit).
        cg_a = C._hash_files(C._CODEGEN_FILES, b"ast:AAAA", b"cgreuse:")
        cg_b = C._hash_files(C._CODEGEN_FILES, b"ast:BBBB", b"cgreuse:")
        assert cg_a != cg_b, "codegen epoch does not depend on the AST epoch (nesting broken)"
        # env_epoch (CACHE-1) folds the codegen epoch — a codegen change re-keys results.
        from TEX_Wrangle import tex_results as R
        assert C.codegen_epoch() in R.env_epoch(), "env_epoch does not carry the codegen epoch"
        r.ok("ast ⊑ codegen ⊑ verdict distinct + nested; env_epoch carries codegen epoch")
    except Exception as e:
        r.fail("CACHE-4 layering", f"{type(e).__name__}: {e}")


def test_cache4_codegen_edit_spares_pkl(r: SubTestResult):
    print("\n--- CACHE-4: the win — a codegen-epoch bump does NOT cold-start the .pkl tier ---")
    from TEX_Wrangle import tex_cache as C
    from TEX_Wrangle.tex_compiler.types import TEXType
    import pickle as _pickle
    try:
        cache = C.get_cache()
        code = "@OUT = @A * 0.7 + 0.1;"
        bt = {"A": TEXType.VEC4}
        fp = cache.fingerprint(code, bt)
        for ext in (".pkl", ".cg"):
            (cache._cache_dir / f"{fp}{ext}").unlink(missing_ok=True)
        prog, *_ = cache.compile_tex(code, bt)          # persists the .pkl under the AST epoch
        pkl = cache._disk_path(fp)
        assert pkl.exists(), "compile_tex did not persist a .pkl"
        rec = _pickle.load(open(pkl, "rb"))
        assert rec["version"] == C.ast_epoch(), ".pkl is not keyed by the AST epoch"
        # Simulate a codegen-only edit: bump the codegen epoch, leave the AST epoch. The .pkl
        # must STILL load (the parse/typecheck/optimize work is not thrown away).
        old_cg = C._CODEGEN_EPOCH
        try:
            C._CODEGEN_EPOCH = "0" * 16
            assert cache._load_from_disk(fp, bt) is not None, \
                "a codegen-epoch bump wrongly cold-started the .pkl tier"
            # and a .cg IS invalidated by that same bump (keyed on the codegen epoch)
            cache._persist_codegen(fp, unsupported=True)   # writes version=old codegen epoch
            C._CODEGEN_EPOCH = "1" * 16
            assert cache._load_codegen_from_disk(fp) is None, \
                "a codegen-epoch bump did not invalidate the .cg sidecar"
        finally:
            C._CODEGEN_EPOCH = old_cg
        r.ok(".pkl survives a codegen bump; .cg is invalidated by it (the CACHE-4 win)")
    except Exception as e:
        r.fail("CACHE-4 codegen-spares-pkl", f"{type(e).__name__}: {e}")


def test_cache4_failsafe_oracle(r: SubTestResult):
    print("\n--- CACHE-4: fail-safe — a reloaded persisted .pkl matches SOURCE ground truth ---")
    # The spot-check that catches a WRONG epoch partition (an AST-file edit that changes the
    # optimized program but bumps no epoch, so a STALE .pkl is served). The ground truth must be
    # INDEPENDENT of the disk cache: parse + typecheck + interpret straight from SOURCE, bypassing
    # get_cache / the .pkl entirely. (An earlier version compiled BOTH sides through the cache —
    # both hit the same .pkl, so it could never detect drift; that was a tautology.) The .cg
    # (codegen-epoch) tier is guarded by the separate codegen==interpreter differential suite.
    from TEX_Wrangle import tex_cache as C, tex_api
    from TEX_Wrangle.tex_compiler.types import TEXType
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    try:
        cache = C.get_cache()
        code = "@OUT = vec4(@A.rgb * 0.5 + 0.2, 1.0);"
        bt = {"A": TEXType.VEC4}
        A = torch.rand(1, 8, 8, 4)
        # GROUND TRUTH — from source, no get_cache(), no .pkl.
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        tm = TypeChecker(binding_types=bt, source=code).check(prog)
        truth = Interpreter().execute(prog, {"A": A.clone()}, tm, output_names=["OUT"],
                                      device="cpu")["OUT"]
        # ARTIFACT PATH — persist, drop the memory tier, reload the .pkl from disk, interpret.
        fp = cache.fingerprint(code, bt)
        (cache._cache_dir / f"{fp}.pkl").unlink(missing_ok=True)
        cache.compile_tex(code, bt)          # persist the (optimized) program
        cache._memory.clear()                # force a disk reload on the next compile
        reloaded = tex_api.execute(tex_api.compile(code, bt), {"A": A.clone()})["OUT"]
        assert torch.allclose(truth, reloaded, atol=1e-6), \
            "the persisted .pkl drifted from source ground truth (a wrong epoch partition)"
        r.ok("persisted .pkl matches source-independent ground truth (real oracle, not a tautology)")
    except Exception as e:
        r.fail("CACHE-4 fail-safe oracle", f"{type(e).__name__}: {e}")


def test_cache1_playhead_keys(r: SubTestResult):
    print("\n--- CACHE-1: every playhead builtin (frame/fps/time) re-keys; no stale serve ---")
    # Regression for the review's CRITICAL: keying only `frame` (and int()-truncating it) let a
    # time- or fps-animation, or a fractional frame, collide onto one key and serve a stale frame.
    # The whole normalized time_context must key, by exact value.
    from TEX_Wrangle import tex_results as R
    # Direct key-level: each playhead component discriminates; fractional frames don't collide.
    try:
        base = dict(program_fp="p", device="cpu", precision="fp32")
        k = R.lineage_key(**base, time_context={"frame": 5.0, "fps": 24.0, "time": 0.0})
        assert R.lineage_key(**base, time_context={"frame": 5.0, "fps": 24.0, "time": 5.0}) != k, \
            "a `time` change did not re-key"
        assert R.lineage_key(**base, time_context={"frame": 5.0, "fps": 60.0, "time": 0.0}) != k, \
            "an `fps` change did not re-key"
        assert R.lineage_key(**base, frame=0.4) != R.lineage_key(**base, frame=0.6), \
            "fractional frames 0.4 and 0.6 collided (int truncation)"
        assert R.lineage_key(**base, frame=-0.5) != R.lineage_key(**base, frame=0.0), \
            "frame -0.5 and 0.0 collided (int truncation)"
        r.ok("time / fps / fractional-frame all discriminate at the key level")
    except Exception as e:
        r.fail("CACHE-1 playhead key discrimination", f"{type(e).__name__}: {e}")

    # End-to-end: a time-driven cook must not serve a stale frame across a time change.
    for dev in _DEVICES:
        try:
            code = "@OUT = vec4(@A.rgb * time, 1.0);"
            A = torch.ones(1, 4, 4, 4)
            a = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                time_context={"frame": 5.0, "fps": 24.0, "time": 0.0})
            b = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                time_context={"frame": 5.0, "fps": 24.0, "time": 3.0})
            assert a.lineage["OUT"] != b.lineage["OUT"], \
                "same frame, different time -> identical key (would serve a stale frame)"
            # and the pixels really do differ, so a collision WOULD have been a wrong serve
            assert not torch.equal(a.outputs["OUT"], b.outputs["OUT"])
            # A Mapping-but-not-dict playhead (a MappingProxyType a host hands out) is duck-typed
            # by the interpreter, so it MUST also key — else it drives pixels but drops from the
            # key (a silent stale serve). Normalized at the prepare() boundary + keyer duck-types.
            from types import MappingProxyType
            m0 = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                 time_context=MappingProxyType({"frame": 5.0, "time": 0.0}))
            m1 = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                 time_context=MappingProxyType({"frame": 5.0, "time": 3.0}))
            assert m0.lineage["OUT"] != m1.lineage["OUT"], \
                "a MappingProxyType playhead drove pixels but fell out of the key (stale serve)"
            r.ok(f"[{dev}] a time-animation (dict + non-dict Mapping) mints distinct keys")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-1 playhead engine", f"{type(e).__name__}: {e}")


def test_cache1_precision_and_batch(r: SubTestResult):
    print("\n--- CACHE-1: re-cook precision label + batch/canvas keying (audit fixes 1,2) ---")
    # Fix 1: when the fp16 finiteness net re-cooks fp32, the frame IS fp32 — the key and
    # CookResult.precision must say fp32, not the original fp16 (else a false cache miss vs the
    # auto-pinned-fp32 cook, and a lie to a host). Unit-test the net's precision report directly
    # (deterministic — no dependence on the auto gate picking fp16).
    try:
        plan = tex_engine.prepare("@OUT = @A * 1.0;", {"A": torch.ones(1, 4, 4, 4)},
                                  precision="fp16", device_mode="cpu")
        bad = {"OUT": torch.full((1, 4, 4, 4), float("inf"))}   # simulate an fp16 overflow
        out, prec = tex_engine._fp16_finiteness_net(bad, True, plan.ctx, plan.tier_id,
                                                    plan.auto_ckey)
        assert prec == "fp32", f"net re-cooked but reported precision {prec!r}"
        assert torch.isfinite(out["OUT"]).all(), "re-cook did not fix the non-finite output"
        # and a no-op (finite / not auto-fp16) reports the original precision
        good = {"OUT": torch.zeros(1, 4, 4, 4)}
        _, prec2 = tex_engine._fp16_finiteness_net(good, False, plan.ctx, plan.tier_id,
                                                   plan.auto_ckey)
        assert prec2 == plan.ctx.eff_precision, "no-op net changed the reported precision"
        r.ok("finiteness net reports the precision it actually cooked at (fp32 on re-cook)")
    except Exception as e:
        r.fail("CACHE-1 re-cook precision", f"{type(e).__name__}: {e}")

    # Precision flows into the key: an fp16 and an fp32 cook of the same inputs key apart.
    try:
        a = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 8, 8, 4)},
                            device_mode="cpu", precision="fp16", want_lineage=True)
        b = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 8, 8, 4)},
                            device_mode="cpu", precision="fp32", want_lineage=True)
        assert a.precision == "fp16" and b.precision == "fp32"
        assert a.lineage["OUT"] != b.lineage["OUT"], "precision did not discriminate the key"
        r.ok("precision discriminates the lineage key end-to-end")
    except Exception as e:
        r.fail("CACHE-1 precision key", f"{type(e).__name__}: {e}")

    # Fix 2: batch keys. Same program, batch-1 vs batch-4 -> DIFFERENT keys (else a frame cache
    # serves a batch-1 frame for a batch-N request).
    for dev in _DEVICES:
        try:
            b1 = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 8, 8, 4)},
                                 device_mode=dev, want_lineage=True)
            b4 = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(4, 8, 8, 4)},
                                 device_mode=dev, want_lineage=True)
            assert b1.outputs["OUT"].shape[0] != b4.outputs["OUT"].shape[0]
            assert b1.lineage["OUT"] != b4.lineage["OUT"], \
                "batch-1 and batch-4 cooks collided on one key (canvas dropped batch)"
            # and a different spatial size still keys apart (canvas carries the full shape)
            s2 = tex_engine.cook("@OUT = @A * 0.5;", {"A": torch.rand(1, 16, 16, 4)},
                                 device_mode=dev, want_lineage=True)
            assert b1.lineage["OUT"] != s2.lineage["OUT"], "spatial size did not re-key"
            r.ok(f"[{dev}] batch + spatial size both discriminate the key")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-1 batch key", f"{type(e).__name__}: {e}")


def test_cache1_pixel_moving_flags(r: SubTestResult):
    print("\n--- CACHE-1: pixel-moving non-binding flags re-key (debug overlay, ic) ---")
    # Audit regression: a flag that MOVES PIXELS but is neither a binding nor part of the output
    # shape would fall out of the lineage key and silent-serve a stale frame across a toggle.
    # Two such flags: debug_nan_highlight (paints magenta/cyan onto raw_output) and
    # latent_channel_count (materializes the program-readable `ic` builtin).

    # (a) debug_nan_highlight paints magenta over NaN. TEX guards every INTERNAL NaN source, so
    # the NaN comes from an input passed through: overlay off -> raw NaN survives; overlay on ->
    # finite magenta paint. Same program/inputs, so ONLY the toggle differs -> keys must differ.
    for dev in _DEVICES:
        try:
            code = "@OUT = @A;"                                   # passthrough: input NaN survives
            A = torch.full((1, 4, 4, 4), float("nan"))
            on = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                 debug_nan_highlight=True)
            off = tex_engine.cook(code, {"A": A.clone()}, device_mode=dev, want_lineage=True,
                                  debug_nan_highlight=False)
            assert torch.isfinite(on.outputs["OUT"]).all(), "overlay-on frame is not painted"
            assert not torch.isfinite(off.outputs["OUT"]).all(), "overlay-off frame is not raw NaN"
            assert on.lineage["OUT"] != off.lineage["OUT"], \
                "debug overlay painted the frame but did not re-key (would serve a stale frame)"
            r.ok(f"[{dev}] debug_nan_highlight discriminates the key (painted != raw)")
        except Exception as e:
            r.fail(f"[{dev}] CACHE-1 debug-overlay key", f"{type(e).__name__}: {e}")

    # (b) latent_channel_count drives `ic`; output shape is channel-count-INDEPENDENT here, so only
    # the key fix keeps a 16-channel cook from serving a 4-channel one.
    try:
        code = "@OUT = vec4(@A.rgb * 0.0 + ic / 16.0, 1.0);"     # pixels = ic/16, shape follows @A
        A = torch.ones(1, 4, 4, 4)
        c16 = tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu", want_lineage=True,
                              latent_channel_count=16)
        c4 = tex_engine.cook(code, {"A": A.clone()}, device_mode="cpu", want_lineage=True,
                             latent_channel_count=4)
        assert list(c16.outputs["OUT"].shape) == list(c4.outputs["OUT"].shape), \
            "test precondition: output shape must be channel-count-independent"
        assert not torch.equal(c16.outputs["OUT"], c4.outputs["OUT"]), "ic did not move pixels"
        assert c16.lineage["OUT"] != c4.lineage["OUT"], \
            "latent_channel_count moved pixels but fell out of the key (stale serve)"
        r.ok("latent_channel_count (`ic`) discriminates the key at equal output shape")
    except Exception as e:
        r.fail("CACHE-1 latent_channel_count key", f"{type(e).__name__}: {e}")


def main():
    r = SubTestResult()
    test_eng12_output_is_born_frozen(r)
    test_eng12_two_strata(r)
    test_eng12_frozen_frame_reenters_scatter(r)
    test_cache1_key_construction(r)
    test_cache1_not_a_content_hash(r)
    test_cache1_engine_integration(r)
    test_cache1_playhead_keys(r)
    test_cache1_precision_and_batch(r)
    test_cache1_pixel_moving_flags(r)
    test_cache2_hit_is_bit_exact(r)
    test_cache2_spill_restore_bit_exact(r)
    test_cache2_verify_drop_and_replace(r)
    test_cache3_warm_state_roundtrip(r)
    test_cache3_version_tag_guard(r)
    test_cache3_prewarm(r)
    test_cache4_epoch_tripwire(r)
    test_cache4_layering(r)
    test_cache4_codegen_edit_spares_pkl(r)
    test_cache4_failsafe_oracle(r)
    return 0 if r.summary() else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
