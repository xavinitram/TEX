"""SEAM-45 — a freeze test for the embedding-host seam (v0.45, "the adoption point").

An embedding host other than ComfyUI adopts a pinned subset of this tree by re-pinning whole
commits; it does not chase HEAD. Before this ask there was no test proving TEX would even
NOTICE if a symbol such a host depends on moved, was renamed, or changed shape — the risk was
silent, and would only surface at the next re-pin, in the worst possible place (a working
integration that stops working with no local diff to blame). This is that test.

It pins, by module path, callable-vs-value kind, and — for every callable — parameter NAMES,
KINDS (positional-or-keyword / keyword-only / var-positional / var-keyword) and default-PRESENCE
(never the default VALUE, which is not part of this contract), every symbol a 2026-09-25 census
of the embedding-host seam found:

  * **Tier 1** (`_TIER1_SPEC`, 118 rows: the census's 114 plus `ResultCache.spill` and
    `Program.time_reads`, added the moment they landed on `main`, plus `cook_observer.register`
    and `.unregister` (OBSERVER-46, v0.46) — see below) — symbols that
    host's own PRODUCT code calls or references. This is the harder promise: these are
    load-bearing for a running integration.
  * **Tier 2** (`_TIER2_SPEC`, 87 rows) — symbols reached ONLY from that host's own tests,
    scripts and benchmarks, never its product code. Still pinned exactly (existence AND
    signature shape), but softer: nothing in the shipped product breaks if one of these moves,
    only that host's own dev-time tooling.

The census found 89 Tier-2 candidates; two do not exist at this head under the exact dotted
path the census recorded (`_KNOWN_PHANTOMS` below) — each is a census resolution artifact or a
host-side monkeypatch fact, not a TEX symbol that was ever shipped and then removed. They are
deliberately NOT pinned (pinning a phantom would just be pinning "AttributeError", which is not
a promise anyone can rely on); `test_seam45_known_phantoms_stay_absent` below still watches them,
so a future change that makes one of them start resolving is a decision this test forces someone
to notice, in either direction.

**Where the tables come from.** They are DATA: a literal, checked-in table generated ONCE, by a
small throwaway script (not part of this tree — it is not a product file, and re-running it
against a moved symbol is exactly the "someone decides on purpose" step this test exists to
force) that walked the census's own symbol list with `inspect.signature` at this ask's base sha
(v0.44.0). Changing a row here is a deliberate edit to a checked-in file, not something a
refactor does by accident.

**Two rows arrived after the census.** `ResultCache.spill` and `Program.time_reads` were the
ASKS lane's own additions, landed on `main` after this census was taken; they are folded
straight into `_TIER1_SPEC` (added post-merge, not re-derived from the whole tree, since they
are two rows and their shape is already known from the code) rather than left uncovered until
the next census. `Program.time_reads` is a dataclass field, not a callable — pinned as a plain
attribute (its default value's type), exactly like any other non-callable row.

**Mutation, both directions**, in `test_seam45_mutation_proves_both_directions`: a renamed
keyword and a removed symbol each red the exact comparator (`_diff`) the two tier tests call —
proved without mutating any real production source, which a test must never do here (invariant
#7 — every change is behaviour-preserving on the default path, and that includes not breaking it
on purpose from inside a test).
"""
from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect

from helpers import SubTestResult

# ── the frozen tables (DATA — generated once from the code at v0.44.0, 9f7c428) ─────────────
#
# key: "<module dotted off TEX_Wrangle>:<symbol>" (a dotted `symbol` walks attributes off the
# module: `ResultCache.put` means `getattr(getattr(mod, "ResultCache"), "put")`; the sentinel
# symbol "__module_itself__" means the module object itself, for a row that is only ever
# referenced, never called).
#
# value: (kind, sig) where kind is one of "class" / "function" / "callable" / "module", or the
# plain runtime type name of a non-callable value ("str", "int", "float", "dict", "set", or an
# enum's own class name).
#
# sig's shape depends on kind, and the split below is the version-proofing rule (SEAM-45b):
# `inspect.signature` is stable across CPython versions ONLY when it reads a real `__code__` --
# a pure-Python function or method (`inspect.isfunction` / `inspect.ismethod`; a `def` reached
# by walking a class also lands here, staticmethod/classmethod already unwrapped by `getattr`).
# Everything else asks `inspect.signature` to render a signature CPython synthesises rather than
# reads, and that rendering has already been proven to change shape between versions (see the
# CLASS case below) -- so the rule pins those by existence and kind only, never a signature:
#
#   * "function": a tuple of (param_name, param_kind, has_default) triples in declaration order.
#     For a method reached by walking the CLASS (not an instance), `self` is left in, exactly as
#     `inspect.signature` reports it that way; for a bound method it is dropped.
#   * "class": `sig` is NEVER an `__init__` signature -- whether a class's `__init__` is its own
#     pure-Python `def` or falls through to a C ancestor (`object.__init__`, `BaseException.__init__`,
#     `Enum`'s machinery) is not something this test can tell apart robustly, and the one caught by
#     CI (`TEXType`, an Enum with no `__init__` of its own) proved `inspect.signature` on that
#     fallback renders a DIFFERENT parameter tuple per interpreter version for the exact same live
#     symbol. So every class is pinned by existence and kind only (`sig=None`), with two additive
#     exceptions whose extra pin is itself version-stable because it never touches a signature:
#       - an Enum subclass pins `("enum_members", (name, ...))` -- `EnumClass.__members__`/
#         iteration order, stable across versions;
#       - a `@dataclass` pins `("dataclass_fields", (name, ...))` -- `dataclasses.fields()`'
#         declaration order, stable across versions and more meaningful to a host than a
#         constructor signature anyway.
#   * "callable": always `sig=None`. Covers builtins and other C-implemented callables
#     (`inspect.isbuiltin` / `inspect.ismethoddescriptor`, e.g. a bound `dict.items` -- CPython
#     3.13 renders it as `()` from its Argument Clinic text signature, 3.11 raises ValueError,
#     "no signature found", for the SAME live symbol) and any other non-function/method callable.
#   * "module": always `sig=None`.

_TIER1_SPEC = {
    'TEX_Wrangle:__version__': ('str', None),
    'tex_api:compile': ('function', (('source', 'POSITIONAL_OR_KEYWORD', False), ('binding_types', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_api:check': ('function', (('source', 'POSITIONAL_OR_KEYWORD', False), ('binding_types', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_api:color_advisories': ('function', (('source', 'POSITIONAL_OR_KEYWORD', False), ('param_values', 'POSITIONAL_OR_KEYWORD', False), ('binding_meta', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_api:prewarm': ('function', (('programs', 'POSITIONAL_OR_KEYWORD', False), ('shapes', 'POSITIONAL_OR_KEYWORD', True), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('compile_mode', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True))),
    'tex_api:LANGUAGE_VERSION': ('str', None),
    'tex_api:Program.time_reads': ('frozenset', None),
    'tex_cache:get_cache': ('function', ()),
    'tex_checkpoint:cook_checkpointed': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('result_cache', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('upstream', 'KEYWORD_ONLY', True), ('cuts', 'KEYWORD_ONLY', True), ('threshold_ms', 'KEYWORD_ONLY', True), ('profile_key', 'KEYWORD_ONLY', True), ('spatial', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True))),
    'tex_checkpoint:materialize': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('result_cache', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('upstream', 'KEYWORD_ONLY', True), ('cuts', 'KEYWORD_ONLY', True), ('threshold_ms', 'KEYWORD_ONLY', True), ('profile_key', 'KEYWORD_ONLY', True), ('spatial', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True))),
    'tex_checkpoint:plan_checkpoints': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('costs', 'KEYWORD_ONLY', True), ('threshold_ms', 'KEYWORD_ONLY', True), ('px', 'KEYWORD_ONLY', True), ('settled', 'KEYWORD_ONLY', True), ('device', 'KEYWORD_ONLY', True))),
    'tex_checkpoint:put_cost_ms': ('function', (('px', 'POSITIONAL_OR_KEYWORD', False), ('device', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_checkpoint:DEFAULT_THRESHOLD_MS': ('float', None),
    'tex_checkpoint:MIN_SAMPLES': ('int', None),
    'tex_compiler.ast_nodes:iter_child_nodes': ('function', (('node', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_compiler.ast_nodes:Assignment': ('class', ('dataclass_fields', ('loc', 'target', 'value', 'op'))),
    'tex_compiler.ast_nodes:BindingRef': ('class', ('dataclass_fields', ('loc', 'name', 'kind', 'type_hint'))),
    'tex_compiler.ast_nodes:FunctionCall': ('class', ('dataclass_fields', ('loc', 'name', 'args'))),
    'tex_compiler.ast_nodes:ParamDecl': ('class', ('dataclass_fields', ('loc', 'name', 'type_hint', 'default_expr', 'metadata'))),
    'tex_compiler.types:TEXType': ('class', ('enum_members', ('INT', 'FLOAT', 'VEC2', 'VEC3', 'VEC4', 'MAT3', 'MAT4', 'STRING', 'ARRAY', 'PLANES', 'VOID'))),
    'tex_compiler.types:TEXType.FLOAT': ('TEXType', None),
    'tex_compiler.types:TEXType.VEC4': ('TEXType', None),
    'tex_cookqueue:CookQueue': ('class', None),
    'tex_cookqueue:Job': ('class', ('dataclass_fields', ('id', 'klass', 'fn', 'reason', 'confidence', 'profile_key', 'px', 'cost_ms', 'score', 'feeds_profile', 'inputs', 'state', 'value', 'error', 'preempt_requested', 'shed_requested', 'resumed', 'attempts', 'preemptions', 'started_at', '_done'))),
    'tex_cookqueue:CLASS_NAMES': ('dict', None),
    'tex_cookqueue:COMMITTED': ('int', None),
    'tex_cookqueue:IDLE_CHECKPOINT': ('str', None),
    'tex_cookqueue:INTERACTIVE': ('int', None),
    'tex_cookqueue:SPECULATIVE': ('int', None),
    'tex_cookqueue:SpeculativePolicy': ('class', None),
    # PACE-45: not a census row (the 2026-09-25 census never covered this class) -- added on
    # its own, deliberately, the moment `done` landed as a new dataclass field on it. See
    # this ask's hand-back for why: `run()`/`cook()`'s return TYPE is exactly as load-bearing
    # for an embedding host as any function this table already pins, and freezing it now
    # (rather than only from the next census onward) means an accidental rename of any of
    # its EXISTING fields reds here immediately instead of at the next census's mercy.
    'tex_engine:CookResult': ('class', ('dataclass_fields', (
        'outputs', 'output_names', 'assigned', 'device', 'precision', 'binding_names',
        'near_singularities', 'lineage', 'out_meta', 'cooked_roi', 'noise_tiers', 'done'))),
    'tex_engine:cook': ('function', (('code', 'POSITIONAL_OR_KEYWORD', False), ('bindings', 'POSITIONAL_OR_KEYWORD', False), ('kwargs', 'VAR_KEYWORD', False))),
    'tex_engine:cook_stage_list': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True))),
    'tex_engine:boundary_lineage_key': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('k', 'POSITIONAL_OR_KEYWORD', False), ('device', 'POSITIONAL_OR_KEYWORD', False), ('precision', 'POSITIONAL_OR_KEYWORD', False), ('upstream', 'KEYWORD_ONLY', False), ('time_context', 'KEYWORD_ONLY', True), ('canvas', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True))),
    'tex_engine:cook_fused_cached': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('k', 'POSITIONAL_OR_KEYWORD', False), ('result_cache', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True), ('upstream', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True))),
    'tex_engine:prepare': ('function', (('code', 'POSITIONAL_OR_KEYWORD', False), ('bindings', 'POSITIONAL_OR_KEYWORD', False), ('chain_payload', 'KEYWORD_ONLY', True), ('device_mode', 'KEYWORD_ONLY', True), ('compile_mode', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('has_latent_input', 'KEYWORD_ONLY', True), ('latent_channel_count', 'KEYWORD_ONLY', True), ('forgive_dead_refs', 'KEYWORD_ONLY', True), ('debug_nan_highlight', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('max_outputs', 'KEYWORD_ONLY', True), ('disown', 'KEYWORD_ONLY', True), ('roi', 'KEYWORD_ONLY', True), ('roi_exec', 'KEYWORD_ONLY', True), ('want_lineage', 'KEYWORD_ONLY', True), ('want_noise_tiers', 'KEYWORD_ONLY', True), ('upstream_keys', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True), ('binding_meta', 'KEYWORD_ONLY', True))),
    'tex_engine:resolve_device': ('function', (('device_mode', 'POSITIONAL_OR_KEYWORD', False), ('bindings', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_engine:run': ('function', (('plan', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_engine:_infer_binding_type': ('function', (('value', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_fusion:detect_fusable_regions': ('function', (('nodes', 'POSITIONAL_OR_KEYWORD', False), ('edges', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_fusion:region_to_collapse_plan': ('function', (('region', 'POSITIONAL_OR_KEYWORD', False), ('node_code', 'POSITIONAL_OR_KEYWORD', False), ('node_params', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_fusion:fused_fingerprint': ('function', (('spec', 'POSITIONAL_OR_KEYWORD', False), ('terminal_code', 'POSITIONAL_OR_KEYWORD', False), ('terminal_bindings', 'POSITIONAL_OR_KEYWORD', False), ('infer_binding_type', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_fusion:is_linear_stage_list': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_fusion:region_to_stages': ('function', (('region', 'POSITIONAL_OR_KEYWORD', False), ('node_code', 'POSITIONAL_OR_KEYWORD', False), ('node_params', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_fusion:_MAX_FUSED_REGION_STAGES': ('int', None),
    'tex_io:BufferDesc': ('class', ('dataclass_fields', ('storage', 'transfer'))),
    'tex_io:decode_to_fp32': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False), ('desc', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_io:__module_itself__': ('module', None),
    'tex_io.exr:read_exr': ('function', (('src', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_io.exr:write_exr': ('function', (('path', 'POSITIONAL_OR_KEYWORD', False), ('pixels', 'POSITIONAL_OR_KEYWORD', False), ('channels', 'KEYWORD_ONLY', True), ('half', 'KEYWORD_ONLY', True), ('compression', 'KEYWORD_ONLY', True))),
    'tex_io.exr:EXRError': ('class', None),
    'tex_io.png:write_png16': ('function', (('path', 'POSITIONAL_OR_KEYWORD', False), ('u16', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_lazy:lazy_required_bindings': ('function', (('code', 'POSITIONAL_OR_KEYWORD', False), ('param_values', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_marshalling:BufferMeta': ('class', ('dataclass_fields', ('colorspace', 'premult', 'frame', 'extra'))),
    'tex_marshalling:get_egress_profile': ('function', ()),
    'tex_marshalling:set_egress_profile': ('function', (('name', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_marshalling:Promise': ('class', None),
    'tex_marshalling:map_inferred_type': ('function', (('inferred', 'POSITIONAL_OR_KEYWORD', False), ('has_latent_input', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_marshalling:merge_buffer_meta': ('function', (('metas', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_marshalling:prepare_output': ('function', (('raw', 'POSITIONAL_OR_KEYWORD', False), ('output_type', 'POSITIONAL_OR_KEYWORD', False), ('profile', 'KEYWORD_ONLY', True))),
    'tex_memory:get_cache_registry': ('function', ()),
    'tex_memory:governor_budget': ('function', (('device', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_memory:active_profile': ('function', ()),
    'tex_memory:profile_knobs': ('function', (('name', 'POSITIONAL_OR_KEYWORD', True),)),
    'tex_memory:estimate_peak_bytes': ('function', (('program', 'POSITIONAL_OR_KEYWORD', False), ('spatial_shape', 'POSITIONAL_OR_KEYWORD', False), ('dtype_bytes', 'POSITIONAL_OR_KEYWORD', True), ('fingerprint', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_memory:register_result_cache': ('function', (('cache', 'POSITIONAL_OR_KEYWORD', False), ('name', 'KEYWORD_ONLY', True), ('evict_order', 'KEYWORD_ONLY', True))),
    'tex_memory:set_profile': ('function', (('name', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_memory:free_tensor_caches': ('function', ()),
    'tex_packing:propagate_quality': ('function', (('own', 'POSITIONAL_OR_KEYWORD', True), ('upstream_qualities', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_packing:PREVIEW': ('str', None),
    'tex_provider:get_provider': ('function', ()),
    'tex_provider:materialize': ('function', (('source_key', 'POSITIONAL_OR_KEYWORD', False), ('t', 'POSITIONAL_OR_KEYWORD', False), ('mode', 'POSITIONAL_OR_KEYWORD', True), ('speculative', 'KEYWORD_ONLY', True))),
    'tex_provider:source_version': ('function', (('source_key', 'POSITIONAL_OR_KEYWORD', False), ('provider', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_provider:stats': ('function', ()),
    'tex_provider:bump_source_version': ('function', (('source_key', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_provider:set_provider': ('function', (('provider', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_provider:declare_window': ('function', (('queue', 'POSITIONAL_OR_KEYWORD', False), ('source_key', 'POSITIONAL_OR_KEYWORD', False), ('t0', 'POSITIONAL_OR_KEYWORD', False), ('t1', 'POSITIONAL_OR_KEYWORD', False), ('confidence', 'KEYWORD_ONLY', True), ('mode', 'KEYWORD_ONLY', True), ('max_frames', 'KEYWORD_ONLY', True))),
    'tex_provider:set_media_budget_mb': ('function', (('mb', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_provider:source_flags': ('function', (('source_keys', 'VAR_POSITIONAL', False),)),
    'tex_recovery:atomic_write': ('function', (('path', 'POSITIONAL_OR_KEYWORD', False), ('write', 'POSITIONAL_OR_KEYWORD', False), ('fsync', 'KEYWORD_ONLY', True))),
    'tex_recovery:sweep_temps': ('function', (('directory', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_results:ResultCache': ('class', None),
    'tex_results:ResultCache.put': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('key', 'POSITIONAL_OR_KEYWORD', False), ('tensor', 'POSITIONAL_OR_KEYWORD', False), ('canvas', 'KEYWORD_ONLY', True), ('quality', 'KEYWORD_ONLY', True), ('storage', 'KEYWORD_ONLY', True), ('kind', 'KEYWORD_ONLY', True), ('home', 'KEYWORD_ONLY', True), ('mask_eligible', 'KEYWORD_ONLY', True))),
    'tex_results:ResultCache.stats': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_results:ResultCache.evict_bytes': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('need', 'POSITIONAL_OR_KEYWORD', False), ('dev_type', 'KEYWORD_ONLY', True), ('playhead', 'KEYWORD_ONLY', True))),
    'tex_results:ResultCache.get': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('key', 'POSITIONAL_OR_KEYWORD', False), ('copy', 'KEYWORD_ONLY', True))),
    'tex_results:ResultCache.governed_bytes': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('dev_type', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_results:ResultCache.patch_region': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('key', 'POSITIONAL_OR_KEYWORD', False), ('patch', 'POSITIONAL_OR_KEYWORD', False), ('window', 'POSITIONAL_OR_KEYWORD', False), ('base', 'KEYWORD_ONLY', True), ('base_key', 'KEYWORD_ONLY', True), ('canvas', 'KEYWORD_ONLY', True), ('quality', 'KEYWORD_ONLY', True), ('storage', 'KEYWORD_ONLY', True))),
    'tex_results:ResultCache.reindex_disk': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_results:ResultCache.spill': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('key', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_results:lineage_key': ('function', (('program_fp', 'KEYWORD_ONLY', False), ('device', 'KEYWORD_ONLY', False), ('precision', 'KEYWORD_ONLY', False), ('params', 'KEYWORD_ONLY', True), ('upstream', 'KEYWORD_ONLY', True), ('frame', 'KEYWORD_ONLY', True), ('time_context', 'KEYWORD_ONLY', True), ('quality', 'KEYWORD_ONLY', True), ('flags', 'KEYWORD_ONLY', True), ('canvas', 'KEYWORD_ONLY', True))),
    'tex_roi:chain_windows': ('function', (('halos', 'POSITIONAL_OR_KEYWORD', False), ('roi', 'POSITIONAL_OR_KEYWORD', False), ('dirty_from', 'POSITIONAL_OR_KEYWORD', True), ('valid', 'POSITIONAL_OR_KEYWORD', True), ('declined', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_roi:covers': ('function', (('valid', 'POSITIONAL_OR_KEYWORD', False), ('needed', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_roi:canonical_roi': ('function', (('roi', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_roi:stage_halo': ('function', (('code', 'POSITIONAL_OR_KEYWORD', False), ('param_values', 'POSITIONAL_OR_KEYWORD', True), ('binding_types', 'POSITIONAL_OR_KEYWORD', True))),
    # OBSERVER-46: the supported cook-observer seam (v0.46) — the alternative to a host
    # monkey-patching `run`/`cook`/`cook_stage_list`/`cook_fused_cached`/`cook_checkpointed`/
    # `boundary_lineage_key`, one of which ROUTE-45 found a re-export + internal self-call
    # already defeats. Not a census row (the census predates this seam); added the moment it
    # landed, same posture as `ResultCache.spill`/`Program.time_reads` above.
    'tex_runtime.cook_observer:register': ('function', (('cb', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_runtime.cook_observer:unregister': ('function', (('handle', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_runtime.host:CookCancelled': ('class', None),
    'tex_runtime.host:NullHostServices': ('class', None),
    'tex_runtime.profile:enabled': ('function', ()),
    'tex_runtime.profile:make_key': ('function', (('program_fp', 'POSITIONAL_OR_KEYWORD', False), ('device_type', 'POSITIONAL_OR_KEYWORD', False), ('precision', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_runtime.profile:reset': ('function', ()),
    'tex_runtime.profile:snapshot': ('function', ()),
    'tex_runtime.profile:enable': ('function', ()),
    'tex_runtime.profile:measure': ('class', None),
    'tex_runtime.profile:predict': ('function', (('key', 'POSITIONAL_OR_KEYWORD', False), ('spatial', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_runtime.profile:samples': ('function', (('key', 'POSITIONAL_OR_KEYWORD', False), ('spatial', 'POSITIONAL_OR_KEYWORD', True), ('need_stages', 'KEYWORD_ONLY', True))),
    'tex_runtime.profile:stage_snapshot': ('function', (('key', 'POSITIONAL_OR_KEYWORD', False), ('spatial', 'POSITIONAL_OR_KEYWORD', True), ('need', 'KEYWORD_ONLY', True))),
    'tex_runtime.profile:__module_itself__': ('module', None),
    'tex_session:default_session': ('function', ()),
    'tex_tool:load_tool': ('function', (('path_or_dict', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_tool:cook_tool': ('function', (('manifest', 'POSITIONAL_OR_KEYWORD', False), ('inputs', 'POSITIONAL_OR_KEYWORD', False), ('params', 'POSITIONAL_OR_KEYWORD', True), ('cook_kwargs', 'VAR_KEYWORD', False))),
    'tex_tool:install_tool': ('function', (('manifest_or_path', 'POSITIONAL_OR_KEYWORD', False), ('dest_dir', 'POSITIONAL_OR_KEYWORD', True), ('warm', 'KEYWORD_ONLY', True), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True))),
    'tex_tool:preflight_tool': ('function', (('manifest', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_tool:tool_summary': ('function', (('manifest', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_tool:validate_manifest': ('function', (('raw', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_tool:warm_tool': ('function', (('manifest', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True))),
    'tex_tool:write_tool': ('function', (('manifest_or_dict', 'POSITIONAL_OR_KEYWORD', False), ('dest_dir', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_tool:TEXToolError': ('class', None),
}

_TIER2_SPEC = {
    'TEX_Wrangle:__file__': ('str', None),
    'tex_api:execute': ('function', (('program', 'POSITIONAL_OR_KEYWORD', False), ('bindings', 'POSITIONAL_OR_KEYWORD', False), ('device', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True), ('output_names', 'KEYWORD_ONLY', True), ('cancel', 'KEYWORD_ONLY', True), ('on_progress', 'KEYWORD_ONLY', True))),
    'tex_api:__module_itself__': ('module', None),
    'tex_cache:TEXCache': ('class', None),
    'tex_cache:TEXCache.compile_ast': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('program', 'POSITIONAL_OR_KEYWORD', False), ('binding_types', 'POSITIONAL_OR_KEYWORD', False), ('source', 'KEYWORD_ONLY', False))),
    'tex_cache:TEXCache.get': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('code', 'POSITIONAL_OR_KEYWORD', False), ('binding_types', 'POSITIONAL_OR_KEYWORD', False), ('fp', 'KEYWORD_ONLY', True))),
    'tex_cache:__file__': ('str', None),
    'tex_checkpoint:_resolve_cuts': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('result_cache', 'POSITIONAL_OR_KEYWORD', False), ('cuts', 'POSITIONAL_OR_KEYWORD', False), ('latent_channel_count', 'KEYWORD_ONLY', False), ('upstream', 'KEYWORD_ONLY', False), ('precision', 'KEYWORD_ONLY', False), ('threshold_ms', 'KEYWORD_ONLY', False), ('profile_key', 'KEYWORD_ONLY', False), ('spatial', 'KEYWORD_ONLY', False), ('device', 'KEYWORD_ONLY', False))),
    'tex_cli:run_program': ('function', (('code', 'POSITIONAL_OR_KEYWORD', False), ('image', 'POSITIONAL_OR_KEYWORD', False), ('device', 'POSITIONAL_OR_KEYWORD', True), ('precision', 'POSITIONAL_OR_KEYWORD', True), ('compile_mode', 'POSITIONAL_OR_KEYWORD', True), ('profile', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_compiler.diagnostics:SourceLoc': ('class', None),
    'tex_compiler.diagnostics:TEXDiagnostic': ('class', ('dataclass_fields', ('code', 'severity', 'message', 'loc', 'source_line', 'end_col', 'suggestions', 'hint', 'docs_url', 'phase'))),
    'tex_compiler.type_checker:BINDING_HINT_TYPES.items': ('callable', None),
    'tex_compiler.type_checker:BINDING_HINT_TYPES': ('dict', None),
    'tex_compiler.types:CHANNEL_MAP': ('dict', None),
    'tex_compiler.types:TEXType.STRING': ('TEXType', None),
    'tex_compiler.types:TEXType.VEC2': ('TEXType', None),
    'tex_compiler.types:TEXType.VEC3': ('TEXType', None),
    'tex_compiler.types:TEXType.VEC3.channels': ('int', None),
    'tex_compiler.types:VALID_SWIZZLES': ('set', None),
    'tex_cookqueue:QueueStats': ('class', ('dataclass_fields', ('submitted', 'completed', 'failed', 'cancelled', 'preempted', 'requeued', 'refused', 'shed', 'preempt_denied', 'waiting'))),
    'tex_cookqueue:CANCELLED': ('str', None),
    'tex_cookqueue:CookQueue.__init__': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('name', 'KEYWORD_ONLY', True), ('min_quantum_ms', 'KEYWORD_ONLY', True), ('max_preemptions', 'KEYWORD_ONLY', True))),
    'tex_cookqueue:CookQueue.submit': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('fn', 'POSITIONAL_OR_KEYWORD', False), ('klass', 'KEYWORD_ONLY', True), ('reason', 'KEYWORD_ONLY', True), ('confidence', 'KEYWORD_ONLY', True), ('profile_key', 'KEYWORD_ONLY', True), ('px', 'KEYWORD_ONLY', True), ('cost_ms', 'KEYWORD_ONLY', True), ('feeds_profile', 'KEYWORD_ONLY', True), ('inputs', 'KEYWORD_ONLY', True))),
    'tex_cookqueue:NEIGHBOR_FRAME': ('str', None),
    'tex_cookqueue:PANEL_OPEN': ('str', None),
    'tex_cookqueue:PENDING': ('str', None),
    'tex_cookqueue:PLAY_HOVER': ('str', None),
    'tex_cookqueue:PREFETCH': ('str', None),
    'tex_cookqueue:RUNNING': ('str', None),
    'tex_cookqueue:WAITING': ('str', None),
    'tex_engine:is_frozen': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_engine:frame_version': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_engine:verify_unmutated': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False), ('stamp', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_engine:__module_itself__': ('module', None),
    'tex_engine:MAX_OUTPUTS': ('int', None),
    'tex_fusion:detect_region_plans': ('function', (('graph', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_fusion:FusionError': ('class', None),
    'tex_fusion:compile_fused': ('function', (('stages', 'POSITIONAL_OR_KEYWORD', False), ('infer_binding_type', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_fusion:prepare_fused': ('function', (('spec', 'POSITIONAL_OR_KEYWORD', False), ('terminal_code', 'POSITIONAL_OR_KEYWORD', False), ('terminal_bindings', 'POSITIONAL_OR_KEYWORD', False), ('infer_binding_type', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_io:encode_from_fp32': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False), ('desc', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_io.exr:__module_itself__': ('module', None),
    'tex_io.png:_chunk': ('function', (('typ', 'POSITIONAL_OR_KEYWORD', False), ('data', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_marshalling:infer_binding_type': ('function', (('value', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_marshalling:__module_itself__': ('module', None),
    'tex_marshalling:_EGRESS': ('dict', None),
    'tex_memory:profiles': ('function', ()),
    'tex_memory:__module_itself__': ('module', None),
    'tex_packing:q8': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_packing:FINAL': ('str', None),
    'tex_provider:get_media_cache': ('function', ()),
    'tex_provider:reset_provider': ('function', ()),
    'tex_provider:provider_id': ('function', (('provider', 'POSITIONAL_OR_KEYWORD', True),)),
    'tex_provider:_normalize': ('function', (('frame', 'POSITIONAL_OR_KEYWORD', False), ('source_key', 'POSITIONAL_OR_KEYWORD', False), ('t', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_provider:quantize_at_rate': ('function', (('t', 'POSITIONAL_OR_KEYWORD', False), ('rate', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_provider:SyntheticFrameProvider': ('class', None),
    'tex_provider:__module_itself__': ('module', None),
    'tex_provider:E_NO_PROVIDER': ('str', None),
    'tex_provider:NullFrameProvider': ('class', None),
    'tex_recovery:Journal': ('class', None),
    'tex_recovery:TMP_PREFIX': ('str', None),
    'tex_runtime.compiled:__module_itself__': ('module', None),
    'tex_runtime.compiled:_try_codegen': ('function', (('program', 'POSITIONAL_OR_KEYWORD', False), ('type_map', 'POSITIONAL_OR_KEYWORD', False), ('fingerprint', 'POSITIONAL_OR_KEYWORD', True), ('_masked_flow', 'KEYWORD_ONLY', True), ('emit_cancel_polls', 'KEYWORD_ONLY', True))),
    'tex_runtime.graphed:__module_itself__': ('module', None),
    'tex_runtime.graphed:GraphedProgram.capture': ('function', (('self', 'POSITIONAL_OR_KEYWORD', False), ('program', 'POSITIONAL_OR_KEYWORD', False), ('bindings', 'POSITIONAL_OR_KEYWORD', False), ('type_map', 'POSITIONAL_OR_KEYWORD', False), ('device', 'POSITIONAL_OR_KEYWORD', False), ('latent_channel_count', 'POSITIONAL_OR_KEYWORD', False), ('output_names', 'POSITIONAL_OR_KEYWORD', False), ('precision', 'POSITIONAL_OR_KEYWORD', False), ('used_builtins', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_runtime.host:get_host_services': ('function', ()),
    'tex_runtime.interpreter:InterpreterError': ('class', None),
    'tex_runtime.profile:disable': ('function', ()),
    'tex_runtime.profile:record': ('function', (('key', 'POSITIONAL_OR_KEYWORD', False), ('ms', 'POSITIONAL_OR_KEYWORD', False), ('spatial', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_runtime.profile:stage_costs': ('function', (('key', 'POSITIONAL_OR_KEYWORD', False), ('spatial', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_runtime.profile:bucket_of': ('function', (('spatial', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_runtime.profile:stage_sink': ('function', ()),
    'tex_runtime.profile:_STATE_MAX': ('int', None),
    'tex_scheduler:SchedNode': ('class', ('dataclass_fields', ('id', 'program_fp', 'spatial_shape', 'precision', 'out_nbytes', 'peak_bytes', 'inputs', 'pin'))),
    'tex_scheduler:_candidates': ('function', (('node', 'POSITIONAL_OR_KEYWORD', False), ('devices', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_scheduler:_toposort': ('function', (('nodes', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_scheduler:plan_placement': ('function', (('nodes', 'POSITIONAL_OR_KEYWORD', False), ('devices', 'KEYWORD_ONLY', True), ('default_device', 'KEYWORD_ONLY', True), ('cook_cost', 'KEYWORD_ONLY', True), ('transfer_cost', 'KEYWORD_ONLY', True), ('previous', 'KEYWORD_ONLY', True), ('hysteresis_ms', 'KEYWORD_ONLY', True))),
    'tex_scheduler:_greedy': ('function', (('order', 'POSITIONAL_OR_KEYWORD', False), ('by_id', 'POSITIONAL_OR_KEYWORD', False), ('cand', 'POSITIONAL_OR_KEYWORD', False), ('default_device', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_scheduler:available_devices': ('function', ()),
    'tex_scheduler:_assignment_cost': ('function', (('order', 'POSITIONAL_OR_KEYWORD', False), ('dev_of', 'POSITIONAL_OR_KEYWORD', False), ('by_id', 'POSITIONAL_OR_KEYWORD', False), ('cook_cost', 'POSITIONAL_OR_KEYWORD', False), ('transfer_cost', 'POSITIONAL_OR_KEYWORD', False), ('greedy_dev', 'POSITIONAL_OR_KEYWORD', False), ('boundary', 'POSITIONAL_OR_KEYWORD', True))),
    'tex_scheduler:_dev_type': ('function', (('device', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_scheduler:_is_linear_chain': ('function', (('order', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_scheduler:graph_from_spec': ('function', (('spec', 'POSITIONAL_OR_KEYWORD', False), ('spatial_shape', 'KEYWORD_ONLY', True), ('precision', 'KEYWORD_ONLY', True))),
    'tex_scheduler:_ENUM_CAP': ('int', None),
    'tex_scheduler:default_transfer_cost': ('function', (('nbytes', 'POSITIONAL_OR_KEYWORD', False), ('src_dev', 'POSITIONAL_OR_KEYWORD', False), ('dst_dev', 'POSITIONAL_OR_KEYWORD', False))),
    'tex_tool:_package_version': ('function', ()),
    'tex_tool:tool_warm_keys': ('function', (('manifest', 'POSITIONAL_OR_KEYWORD', False),)),
    'tex_tool:_META_KEYS': ('set', None),
}

#: The census's remaining Tier-2 rows (89 total; 87 above) that do NOT exist at this head under
#: the exact dotted path recorded, each with why and its current equivalent (if any).
#: `test_seam45_known_phantoms_stay_absent` keeps watching both so a future change that makes one
#: of them resolve is noticed rather than silently absorbed.
_KNOWN_PHANTOMS = {
    'tex_cookqueue:CookQueue.INTERACTIVE':
        "no such class attribute -- INTERACTIVE is module-level only "
        "('tex_cookqueue:INTERACTIVE' above, Tier 1); CookQueue never re-declares it as its own",
    'tex_engine:run.__wrapped__':
        "tex_engine.run carries no __wrapped__ at this head (no functools.wraps anywhere in "
        "this tree outside tests/). No TEX-side equivalent is owed: an embedding host that "
        "wraps run() with its own functools.wraps-based patch and reads .__wrapped__ back off "
        "its OWN wrapper is reading an attribute IT installed, not one TEX ships",
}


# ── the comparator ───────────────────────────────────────────────────────────────────────────

def _resolve_from(mod, symbol: str):
    """Attribute-walk `symbol` (dot-separated) off an already-imported module object.

    Split out from `_resolve` below so a mutation test can hand this a synthetic double
    without touching `sys.modules` or any real TEX module."""
    if symbol == "__module_itself__":
        return True, mod
    obj = mod
    for part in symbol.split("."):
        if not hasattr(obj, part):
            return False, None
        obj = getattr(obj, part)
    return True, obj


def _resolve(dotted_module: str, symbol: str):
    mod = importlib.import_module(
        "TEX_Wrangle" if dotted_module == "TEX_Wrangle" else "TEX_Wrangle." + dotted_module)
    return _resolve_from(mod, symbol)


def _sig_tuple(sig: inspect.Signature, skip_self: bool = False):
    out = []
    for p in sig.parameters.values():
        if skip_self and p.name == "self":
            continue
        out.append((p.name, str(p.kind), p.default is not inspect.Parameter.empty))
    return tuple(out)


def _describe(obj):
    """(kind, sig) exactly in the frozen tables' shape -- see the module docstring.

    SEAM-45b: a class is NEVER described by its `__init__` signature -- see the module-level
    comment above `_TIER2_SPEC` for why (`inspect.signature` on a class's `__init__` is not
    version-stable whenever that `__init__` is not the class's own pure-Python `def`, and this
    function has no robust way to tell that apart from here). An Enum or a dataclass instead
    pins the one thing about its shape that IS version-stable and does not touch a signature."""
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            return "class", ("enum_members", tuple(m.name for m in obj))
        if dataclasses.is_dataclass(obj):
            return "class", ("dataclass_fields", tuple(f.name for f in dataclasses.fields(obj)))
        return "class", None
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return "function", None
        return "function", _sig_tuple(sig, skip_self=inspect.ismethod(obj))
    if isinstance(obj, type(inspect)):   # a module object, of any module
        return "module", None
    if callable(obj) and not isinstance(obj, type):
        # Anything reaching here is not a pure-Python function/method (that branch is above):
        # a builtin, a method descriptor, or any other C-implemented or non-`def`-backed
        # callable. None of those have a version-stable `inspect.signature` (see the module
        # docstring), so this is pinned by existence and kind only.
        return "callable", None
    return type(obj).__name__, None


def _diff(key: str, expected):
    """`None` if the live symbol `key` ("module:symbol") matches frozen `expected` exactly;
    otherwise one line describing the mismatch. This is the exact function both tier tests
    call, and the one `test_seam45_mutation_proves_both_directions` exercises directly."""
    modname, _, symbol = key.partition(":")
    ok, obj = _resolve(modname, symbol)
    if not ok:
        return f"{key}: no longer exists"
    kind, sig = _describe(obj)
    exp_kind, exp_sig = expected
    if kind != exp_kind:
        return f"{key}: kind changed {exp_kind!r} -> {kind!r}"
    if sig != exp_sig:
        return f"{key}: signature changed\n      was: {exp_sig}\n      now: {sig}"
    return None


# ── the keywords tex_engine.cook's **kwargs path documents (the host stresses this) ─────────

#: `tex_engine.cook(code, bindings, **kwargs)` forwards straight to `prepare()`; this is
#: `prepare`'s own keyword-only parameter set, computed FROM the frozen tier-1 row above rather
#: than typed out a second time (so the two can't drift against each other by hand-edit).
_COOK_DOCUMENTED_KWARGS = frozenset(
    name for (name, kind, _default) in _TIER1_SPEC["tex_engine:prepare"][1]
    if kind == "KEYWORD_ONLY")


# ── the tests ─────────────────────────────────────────────────────────────────────────────────

def test_seam45_tier1_host_seam_pinned(r: SubTestResult):
    """Tier 1: every symbol the census found the embedding-host's PRODUCT code calling or
    referencing (114 census rows plus 2 landed after it: `ResultCache.spill`,
    `Program.time_reads`) exists at this head with exactly the frozen kind and signature
    shape. This is the harder promise -- a mismatch here is load-bearing for a running
    integration, not just its dev tooling."""
    print("\n--- SEAM-45: Tier 1 (the embedding-host product seam) ---")
    mismatches = [d for d in (_diff(k, v) for k, v in sorted(_TIER1_SPEC.items())) if d]
    if mismatches:
        r.fail("SEAM-45 tier-1 seam", "\n" + "\n".join(mismatches))
    else:
        r.ok(f"all {len(_TIER1_SPEC)} tier-1 (product) seam symbols unchanged")


def test_seam45_tier2_test_only_seam_pinned(r: SubTestResult):
    """Tier 2: every symbol the census found reachable ONLY from that host's own tests,
    scripts and benchmarks (87 of 89 rows; the other two are documented phantoms, see
    `test_seam45_known_phantoms_stay_absent`). Pinned exactly, but softer -- nothing in the
    shipped product depends on these, only that host's own dev-time tooling."""
    print("\n--- SEAM-45: Tier 2 (the test-only seam) ---")
    mismatches = [d for d in (_diff(k, v) for k, v in sorted(_TIER2_SPEC.items())) if d]
    if mismatches:
        r.fail("SEAM-45 tier-2 seam", "\n" + "\n".join(mismatches))
    else:
        r.ok(f"all {len(_TIER2_SPEC)} tier-2 (test-only) seam symbols unchanged")


def test_seam45_known_phantoms_stay_absent(r: SubTestResult):
    """The two Tier-2 census rows that never resolved at this head. Not pinned as real rows
    (pinning "AttributeError" is not a promise), but watched: if one of them starts resolving,
    that is a decision someone should notice and move into `_TIER2_SPEC` on purpose, exactly
    as this whole test exists to force for every other row."""
    print("\n--- SEAM-45: the census rows that were phantoms at this head ---")
    resurfaced = []
    for key in _KNOWN_PHANTOMS:
        modname, _, symbol = key.partition(":")
        ok, _obj = _resolve(modname, symbol)
        if ok:
            resurfaced.append(key)
    if resurfaced:
        r.fail("SEAM-45 phantom census rows",
               f"now resolve and should move into _TIER2_SPEC on purpose: {resurfaced}")
    else:
        r.ok(f"{len(_KNOWN_PHANTOMS)} known-phantom census rows are still absent, as documented")


def test_seam45_cook_documented_kwargs_path_pinned(r: SubTestResult):
    """`tex_engine.cook(code, bindings, **kwargs)` forwards every kwarg straight through to
    `prepare()` via `run(prepare(code, bindings, **kwargs))` -- the host relies on that
    forwarding to reach `time_context=`, `compile_mode=`, and every other keyword `prepare()`
    takes, without `cook` ever naming one itself. Pins the forwarding shape AND the keyword
    set it reaches, derived from the code (`prepare`'s own signature) rather than the
    docstring's prose."""
    print("\n--- SEAM-45: tex_engine.cook's documented **kwargs path ---")
    import TEX_Wrangle.tex_engine as tex_engine

    body = inspect.getsource(tex_engine.cook)
    stripped = "".join(body.split())
    if "returnrun(prepare(code,bindings,**kwargs))" in stripped:
        r.ok("tex_engine.cook still forwards **kwargs to prepare() via run(prepare(...))")
    else:
        r.fail("SEAM-45 cook forwarding", f"cook() no longer forwards this way:\n{body}")

    live_kwonly = {name for name, p in inspect.signature(tex_engine.prepare).parameters.items()
                   if p.kind == inspect.Parameter.KEYWORD_ONLY}
    if live_kwonly == _COOK_DOCUMENTED_KWARGS:
        r.ok(f"cook's documented **kwargs path is exactly {sorted(_COOK_DOCUMENTED_KWARGS)}")
    else:
        r.fail("SEAM-45 cook kwargs",
               f"live prepare() kw-only names drifted from the frozen list: "
               f"missing={sorted(_COOK_DOCUMENTED_KWARGS - live_kwonly)} "
               f"extra={sorted(live_kwonly - _COOK_DOCUMENTED_KWARGS)}")


def test_seam45_mutation_proves_both_directions(r: SubTestResult):
    """The freeze test's whole job is to red on these two shapes. Proved directly against
    `_diff` -- the exact comparator the two tier tests above call -- rather than a parallel
    synthetic path that might drift from what the real tests do. Production code is never
    mutated (invariant #7); only a COPY of a frozen row's expected shape is perturbed.

    Direction 1 (a renamed keyword): copy one frozen row's expected shape, rename one keyword
    in the copy, and confirm `_diff` still reds the LIVE symbol against that copy -- proving a
    kwarg rename (in either direction, code or table) cannot pass silently.
    Direction 2 (a removed symbol): reuse a census row already proven absent at this head
    (`_KNOWN_PHANTOMS`) and confirm `_diff` reports "no longer exists" for it, not a silent
    pass."""
    print("\n--- SEAM-45 mutation: prove both directions ---")

    key = "tex_engine:boundary_lineage_key"
    real_kind, real_sig = _TIER1_SPEC[key]
    renamed_sig = tuple(
        (name + "_renamed" if name == "upstream" else name, kind, has_default)
        for (name, kind, has_default) in real_sig)
    verdict = _diff(key, (real_kind, renamed_sig))
    if verdict is None:
        r.fail("SEAM-45 mutation (rename)",
               "a renamed keyword in the frozen table did not red against the real symbol")
    else:
        r.ok("a renamed keyword reds `_diff` -- " + verdict.splitlines()[0])

    phantom_key = next(iter(_KNOWN_PHANTOMS))
    verdict = _diff(phantom_key, ("function", ()))
    if verdict is None or "no longer exists" not in verdict:
        r.fail("SEAM-45 mutation (removal)",
               f"a removed symbol ({phantom_key}) did not red as 'no longer exists': {verdict!r}")
    else:
        r.ok(f"a removed symbol reds `_diff` -- {verdict}")


def test_seam45b_class_pinning_mutation_proves_both_directions(r: SubTestResult):
    """SEAM-45b's own mutation proof: a class is pinned by existence/kind plus, for an Enum or
    a dataclass, its member/field NAMES (never an `__init__` signature -- see the module
    docstring). This function proves that new pin still catches a rename in either direction,
    the same way `test_seam45_mutation_proves_both_directions` proves it for a function's
    keyword. Again: a COPY of a frozen row's expected shape is perturbed, never the real
    symbol (invariant #7).

    Direction 1 (a renamed Enum member): copy `TEXType`'s frozen `enum_members` tuple, rename
    one member in the copy, and confirm `_diff` reds the live `TEXType` against it.
    Direction 2 (a renamed dataclass field): copy `Job`'s frozen `dataclass_fields` tuple,
    rename one field in the copy, and confirm `_diff` reds the live `Job` against it."""
    print("\n--- SEAM-45b mutation: class pinning, both directions ---")

    enum_key = "tex_compiler.types:TEXType"
    enum_kind, (enum_tag, enum_members) = _TIER1_SPEC[enum_key]
    renamed_members = tuple(
        name + "_RENAMED" if name == "VOID" else name for name in enum_members)
    verdict = _diff(enum_key, (enum_kind, (enum_tag, renamed_members)))
    if verdict is None:
        r.fail("SEAM-45b mutation (enum member rename)",
               "a renamed Enum member in the frozen table did not red against the real symbol")
    else:
        r.ok("a renamed Enum member reds `_diff` -- " + verdict.splitlines()[0])

    dc_key = "tex_cookqueue:Job"
    dc_kind, (dc_tag, dc_fields) = _TIER1_SPEC[dc_key]
    renamed_fields = tuple(
        name + "_renamed" if name == "profile_key" else name for name in dc_fields)
    verdict = _diff(dc_key, (dc_kind, (dc_tag, renamed_fields)))
    if verdict is None:
        r.fail("SEAM-45b mutation (dataclass field rename)",
               "a renamed dataclass field in the frozen table did not red against the real "
               "symbol")
    else:
        r.ok("a renamed dataclass field reds `_diff` -- " + verdict.splitlines()[0])
