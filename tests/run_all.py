#!/usr/bin/env python
"""Standalone TEX test runner — python tests/run_all.py"""
# CACHE-0: point the disk cache at a scratch dir BEFORE any TEX import — get_cache()
# resolves the location once, on first call — so a test run never writes compiled
# artifacts into the shipping package's .tex_cache. setdefault: an outer harness that
# already chose a dir wins.
import os as _os
import tempfile as _tempfile
_os.environ.setdefault(
    "TEX_CACHE_DIR", _os.path.join(_tempfile.gettempdir(), "tex_test_cache"))

from helpers import SubTestResult

from test_lexer import test_lexer, test_lexer_locations, test_lexer_v11
from test_parser import (
    test_parser, test_parser_v11, test_parser_lvalue_clone,
    test_array_decl_no_hang,
)
from test_type_checker import test_type_checker, test_stdlib_promote_typing
from test_interpreter import (
    test_interpreter, test_for_loops, test_break_continue,
    test_while_loops, test_compound_assignments,
)
from test_language import (
    test_channel_assignment, test_output_types, test_if_without_else,
    test_swizzle_patterns, test_else_if_chains, test_ternary_exhaustive,
    test_scope_and_shadowing, test_operator_edge_cases, test_casting_exhaustive,
    test_vec2_type,
)
from test_diagnostics import test_error_paths, test_diagnostic_quality
from test_performance import test_performance
from test_stdlib import (
    test_stdlib_coverage, test_stdlib_extended, test_numerical_edge_cases,
    test_new_stdlib_functions, test_missing_stdlib_functions,
    test_numeric_edge_case_matrix, test_nan_inf_propagation,
    test_image_reductions, test_sdf_functions, test_new_builtins_and_fixes,
    test_stdlib_edge_cases, test_stdlib_nan_inf,
)
from test_strings_arrays import (
    test_string, test_string_functions_v04, test_string_edge_cases,
    test_arrays, test_vec_arrays, test_string_arrays, test_array_bounds,
)
from test_noise_sampling import (
    test_noise, test_new_noise_functions, test_3d_noise,
    test_arithmetic_hash_noise, test_sampling, test_sample_mip,
    test_gauss_blur_and_mip_gauss,
)
from test_bindings_params import (
    test_named_bindings, test_binding_access, test_binding_access_advanced,
    test_scatter_writes, test_wireable_params, test_new_param_types,
    test_user_functions, test_user_functions_advanced,
)
from test_integration import (
    test_examples, test_example_files, test_example_files_compiled,
    test_cache, test_cache_eviction,
    test_device_selection, test_torch_compile, test_is_changed_hash,
    test_batch_temporal, test_latent, test_auto_inference, test_v03_features,
    test_realistic_sizes, test_matrix_types, test_matrix_benchmarks,
    test_node_helpers, test_compiled_audit_fixes, test_fusion_memo,
)
from test_codegen_optimizer import (
    test_codegen_equivalence, test_optimization_regressions, test_licm,
    test_optimizer_passes, test_optimizer_type_consistency,
    test_optimizer_isint_unary, test_optimizer_pure_fn_cse_licm,
    test_optimizer_dce_side_effects, test_codegen_audit_fixes,
    test_codegen_sample_hoist_in_branches,
)
from test_aliasing_cow import (
    test_cow_channel_array_writes, test_cow_binding_and_function_holes,
    test_literal_cache_persistence, test_scatter_ownership,
    test_clamp_and_gridbuf, test_fp16_guards, test_noise_backend_gate,
)
from test_v015_phase0 import (
    test_pc1_inductor_cache_dir, test_oom_detection,
    test_sample_mip_inference_tensor, test_uc5_literal_array_index,
)
from test_v015_phase1 import (
    test_pc3_codegen_persistence, test_ct1_fused_disk_persistence,
    test_pc2_precompile_safety,
)
from test_v015_phase2 import (
    test_q2_purity_dce, test_uc3_uniform_loop, test_uc2_stencil_routing,
    test_uc1_cuda_graph, test_uc4_const_prop, test_q1_fused_capture,
)
from test_v015_phase3 import (
    test_m1_peak_estimator, test_m1_free_caches, test_m2_cache_budget,
    test_m3_fp16_mode, test_m4_tiling, test_m5_out_reuse,
)
from test_v015_phase4 import test_q4_stage_attribution, test_ct2_offset_sourceloc
from test_v015_phase5 import (
    test_cc2_state_machine, test_cc2_no_stall_sim, test_cc2_end_to_end,
    test_q3_fusion_widening, test_q5_chain_preflight, test_q6_preview_downscale,
)
from test_v015_audit_fixes import (
    test_uc3_fractional_and_bindingmut, test_uc4_array_shadow_constprop,
    test_q5_preflight_from_spec, test_q6_preview_kwarg_popped,
    test_m1_oom_unwrap, test_m3_fp16_reconcile, test_uc2_stencil_exact_only,
    test_m4_tiling_guards, test_uc1_graph_vec_param, test_cc1_triton_hint,
    test_p2_cache_hygiene, test_p2_tap_cap, test_mem1_evict_preserves_graphs,
    test_p2_pc2_scoped_deletion, test_p2_pc1_sibling_sweep,
)
from test_failure_modes import (
    test_fm_class_a_auto_lifecycle, test_fm_class_b_restart,
    test_fm_class_c_entrypoint, test_fm_class_d_cross_tier,
    test_fm_class_e_fp16_sweep,
)
from test_v016_phase1 import test_pf1_pf2_graph_gate, test_static_gate_noise
from test_v016_phase2 import (
    test_uc4_neg_const_prop, test_m5_int_binding, test_m2cpu_and_m1_freeretry,
)
from test_v016_phase4 import (
    test_sl3_color_management, test_sl1_compositing, test_sl2_blend_modes,
    test_sl4_morphology, test_lx8_const_arrays, test_lx9_self_swizzle_write,
    test_sl2_fp16_divide_guard,
)
from test_no_numpy_ban import test_no_numpy_ban
from test_release_gate import test_version_consistency, test_codegen_determinism, test_scatter_determinism_band
from test_v017_phase1 import (
    test_tst5_tier_trace,
    test_tst6_registry_parity,
    test_tst2_edge_matrix,
    test_tst4_operator_completeness,
    test_tst1_differential_fuzzer, test_a1_1_auto_precision_fuzz,
    test_tst7_runner_coverage,
)
from test_v017_phase2 import (
    test_reg1_registry_parity, test_tst3_taxonomy_consistency, test_doc4_reference,
    test_doc5_examples_index, test_reg2_loc_budget,
)
from test_v017_phase3 import (
    test_str5_passes_order, test_str6_emit_dispatch_registry,
    test_str7_codegen_split, test_str9_stmt_dispatch, test_str4_write_collectors,
    test_str2_select_tier_matrix, test_c2_clamp_mixed_bounds,
)
from test_cross_device_envelope import test_prlp1_cross_device_envelope
from test_determinism_pin import test_prlp5_determinism_pin
from test_v018_docs import test_doc7b_map_drift, test_reg1b_doc_ex_populated, test_c5ux_no_render_overstatement, test_c6st_cache_count_agree, test_c3ux_error_codes_resolve
from test_v019_phase1 import (test_a1_2_fusion_lazy_precision_tiers, test_c2st_fp16_taxonomy_federated, test_c3st_gm_rules, test_c1st_execute_line_budget, test_c4st_js_loc_ratchet, test_a1_6_cli_argv)
from test_v019_phase2 import (test_s1_core_no_comfy, test_s1_comfyui_free_execution,
    test_s5_arch_caveat, test_s5_doctor_carries_caveat, test_s4_validate_hw_cli,
    test_s4_validate_hw_runs, test_s4_validate_hw_console_cp1252_safe)
from test_v018_precision import (
    test_prlp4_fp16_safe_reductions, test_prlp4_arr_reductions_fp16_safe,
    test_prlp2_auto_gate, test_prlp2_fp16_accuracy_fuzzer, test_prlp2_node_path_perf,
    test_c1_amplification_gate, test_c2_data_dependent_nan, test_c2_finiteness_net_recovers,
)
from test_v018_memory import (
    test_mem2_pool_trim_gating, test_mem3_fp16_estimator, test_mem4_per_device_budget,
)
from test_v018_ux import (
    test_dbg1_perf_hud_payload, test_ux1_diagnostics_reachability,
    test_dbg3_nan_overlay, test_lx5_debug_print, test_dbg4_doctor,
    test_ux2_tooltip_honesty, test_lx5_json_nan_safe, test_dbg1_nan_fingerprint,
)
from test_v018_portability import (
    test_port1_import_lint, test_port1_host_services, test_port2_facade,
    test_port2_program_shape, test_port3_cli, test_port3_cli_edges, test_hw2_multi_gpu_device_context,
    test_hw4_cpu_threads, test_port3_16bit_png,
)
from test_v018_phase4 import (
    test_prlp6_tf32_profile, test_hw3_triton_validation_skips,
    test_hw1_pf1_calibration_smoke,
)
from test_lazy_cooking import (
    test_lazy_analysis, test_lazy_check_status, test_lazy_execute_path,
    test_lazy_schema_pool_ci,
)


def main():
    print("=" * 60)
    print("TEX Test Suite")
    print("=" * 60)

    r = SubTestResult()

    test_lexer(r)
    test_lexer_locations(r)
    test_lexer_v11(r)
    test_parser(r)
    test_parser_v11(r)
    test_parser_lvalue_clone(r)
    test_array_decl_no_hang(r)
    test_type_checker(r)
    test_stdlib_promote_typing(r)
    test_interpreter(r)
    test_for_loops(r)
    test_break_continue(r)
    test_compound_assignments(r)
    test_examples(r)
    test_example_files(r)
    test_example_files_compiled(r)
    test_cache(r)
    test_device_selection(r)
    test_torch_compile(r)
    test_sampling(r)
    test_noise(r)
    test_channel_assignment(r)
    test_output_types(r)
    test_error_paths(r)
    test_if_without_else(r)
    test_stdlib_coverage(r)
    test_stdlib_extended(r)
    test_numerical_edge_cases(r)
    test_is_changed_hash(r)
    test_swizzle_patterns(r)
    test_performance(r)
    test_cache_eviction(r)
    test_latent(r)
    test_string(r)
    test_named_bindings(r)
    test_arrays(r)
    test_auto_inference(r)
    test_batch_temporal(r)
    test_vec_arrays(r)
    test_string_arrays(r)
    test_image_reductions(r)
    test_matrix_types(r)
    test_matrix_benchmarks(r)
    test_v03_features(r)
    test_wireable_params(r)
    test_new_param_types(r)
    test_string_functions_v04(r)
    test_while_loops(r)
    test_new_stdlib_functions(r)
    test_else_if_chains(r)
    test_optimization_regressions(r)
    test_diagnostic_quality(r)
    test_ternary_exhaustive(r)
    test_user_functions_advanced(r)
    test_binding_access_advanced(r)
    test_scope_and_shadowing(r)
    test_operator_edge_cases(r)
    test_casting_exhaustive(r)
    test_user_functions(r)
    test_binding_access(r)
    test_missing_stdlib_functions(r)
    test_numeric_edge_case_matrix(r)
    test_array_bounds(r)
    test_string_edge_cases(r)
    test_realistic_sizes(r)
    test_nan_inf_propagation(r)
    test_codegen_equivalence(r)
    test_scatter_writes(r)
    test_cow_channel_array_writes(r)
    test_cow_binding_and_function_holes(r)
    test_literal_cache_persistence(r)
    test_scatter_ownership(r)
    test_clamp_and_gridbuf(r)
    test_fp16_guards(r)
    test_noise_backend_gate(r)
    test_pc1_inductor_cache_dir(r)
    test_oom_detection(r)
    test_sample_mip_inference_tensor(r)
    test_uc5_literal_array_index(r)
    test_pc3_codegen_persistence(r)
    test_pc2_precompile_safety(r)
    test_ct1_fused_disk_persistence(r)
    test_q2_purity_dce(r)
    test_uc4_const_prop(r)
    test_uc1_cuda_graph(r)
    test_q1_fused_capture(r)
    test_m1_peak_estimator(r)
    test_m1_free_caches(r)
    test_m2_cache_budget(r)
    test_m3_fp16_mode(r)
    test_m4_tiling(r)
    test_m5_out_reuse(r)
    test_q4_stage_attribution(r)
    test_ct2_offset_sourceloc(r)
    test_cc2_state_machine(r)
    test_cc2_no_stall_sim(r)
    test_cc2_end_to_end(r)
    test_q3_fusion_widening(r)
    test_q5_chain_preflight(r)
    test_q6_preview_downscale(r)
    test_uc2_stencil_routing(r)
    test_uc3_uniform_loop(r)
    test_vec2_type(r)
    test_arithmetic_hash_noise(r)
    test_new_noise_functions(r)
    test_3d_noise(r)
    test_sdf_functions(r)
    test_sample_mip(r)
    test_gauss_blur_and_mip_gauss(r)
    test_new_builtins_and_fixes(r)
    test_licm(r)
    test_stdlib_edge_cases(r)
    test_stdlib_nan_inf(r)
    test_optimizer_passes(r)
    test_optimizer_type_consistency(r)
    test_optimizer_isint_unary(r)
    test_optimizer_pure_fn_cse_licm(r)
    test_optimizer_dce_side_effects(r)
    test_codegen_audit_fixes(r)
    test_codegen_sample_hoist_in_branches(r)
    test_compiled_audit_fixes(r)
    test_fusion_memo(r)
    test_node_helpers(r)
    test_uc3_fractional_and_bindingmut(r)
    test_uc4_array_shadow_constprop(r)
    test_q5_preflight_from_spec(r)
    test_q6_preview_kwarg_popped(r)
    test_m1_oom_unwrap(r)
    test_m3_fp16_reconcile(r)
    test_uc2_stencil_exact_only(r)
    test_m4_tiling_guards(r)
    test_uc1_graph_vec_param(r)
    test_cc1_triton_hint(r)
    test_p2_cache_hygiene(r)
    test_p2_tap_cap(r)
    test_mem1_evict_preserves_graphs(r)
    test_p2_pc2_scoped_deletion(r)
    test_p2_pc1_sibling_sweep(r)
    test_fm_class_a_auto_lifecycle(r)
    test_fm_class_b_restart(r)
    test_fm_class_c_entrypoint(r)
    test_fm_class_d_cross_tier(r)
    test_fm_class_e_fp16_sweep(r)
    test_pf1_pf2_graph_gate(r)
    test_static_gate_noise(r)
    test_uc4_neg_const_prop(r)
    test_m5_int_binding(r)
    test_m2cpu_and_m1_freeretry(r)
    test_sl3_color_management(r)
    test_sl1_compositing(r)
    test_sl2_blend_modes(r)
    test_sl4_morphology(r)
    test_lx8_const_arrays(r)
    test_lx9_self_swizzle_write(r)
    test_sl2_fp16_divide_guard(r)
    test_no_numpy_ban(r)
    test_version_consistency(r)
    test_codegen_determinism(r)
    test_tst5_tier_trace(r)
    test_tst6_registry_parity(r)
    test_tst2_edge_matrix(r)
    test_tst4_operator_completeness(r)
    test_tst1_differential_fuzzer(r)
    test_a1_1_auto_precision_fuzz(r)
    test_tst7_runner_coverage(r)
    test_reg1_registry_parity(r)
    test_tst3_taxonomy_consistency(r)
    test_doc4_reference(r)
    test_doc5_examples_index(r)
    test_reg2_loc_budget(r)
    test_str5_passes_order(r)
    test_str6_emit_dispatch_registry(r)
    test_str7_codegen_split(r)
    test_str9_stmt_dispatch(r)
    test_str4_write_collectors(r)
    test_str2_select_tier_matrix(r)
    test_c2_clamp_mixed_bounds(r)

    # v0.18.0 Phase 0 — stability pins + doc integrity
    test_prlp1_cross_device_envelope(r)
    test_prlp5_determinism_pin(r)
    test_scatter_determinism_band(r)  # A1-4: reads the pin's recorded value; must follow it
    test_doc7b_map_drift(r)
    test_reg1b_doc_ex_populated(r)
    test_c5ux_no_render_overstatement(r)
    test_c6st_cache_count_agree(r)
    test_c3ux_error_codes_resolve(r)
    test_a1_2_fusion_lazy_precision_tiers(r)
    test_c2st_fp16_taxonomy_federated(r)
    test_c3st_gm_rules(r)
    test_c1st_execute_line_budget(r)
    test_c4st_js_loc_ratchet(r)
    test_a1_6_cli_argv(r)
    test_s1_core_no_comfy(r)
    test_s1_comfyui_free_execution(r)
    test_s5_arch_caveat(r)
    test_s5_doctor_carries_caveat(r)
    test_s4_validate_hw_cli(r)
    test_s4_validate_hw_runs(r)
    test_s4_validate_hw_console_cp1252_safe(r)

    from test_v019_phase3 import (test_c6ux_default_code_snippet_hint, test_s3_cheatsheet_drift,
        test_s3_worked_examples_compile, test_s2_workflows_smoke, test_s2_workflows_drift,
        test_c4ux_cyan_on_singularity, test_c4ux_clean_no_cyan, test_c4ux_additive_and_zero_cost_off,
        test_c1ux_c2ux_frontend_present, test_c2ux_doctor_payload_shape)
    test_c6ux_default_code_snippet_hint(r)
    test_s3_cheatsheet_drift(r)
    test_s3_worked_examples_compile(r)
    test_s2_workflows_smoke(r)
    test_s2_workflows_drift(r)
    test_c4ux_cyan_on_singularity(r)
    test_c4ux_clean_no_cyan(r)
    test_c4ux_additive_and_zero_cost_off(r)
    test_c1ux_c2ux_frontend_present(r)
    test_c2ux_doctor_payload_shape(r)

    from test_v019_phase4 import (test_p3_matvec_interp_codegen_bit_exact,
        test_p3_cuda_matches_matmul_within_ulp, test_p3_cpu_keeps_matmul,
        test_p4_tile_safe_memo, test_p4_memo_key_is_cook_fingerprint,
        test_p2_noise_compile_dynamic, test_p6_noise_compile_visibility,
        test_a5_1_reserved_word_hints, test_a5_2_recursive_examples,
        test_f5_codegen_scalar_loop_no_crash, test_f5b_lerp_family_fused_bit_exact,
        test_f5c_pow_mod_codegen_fidelity)
    test_p3_matvec_interp_codegen_bit_exact(r)
    test_p3_cuda_matches_matmul_within_ulp(r)
    test_p3_cpu_keeps_matmul(r)
    test_p4_tile_safe_memo(r)
    test_p4_memo_key_is_cook_fingerprint(r)
    test_p2_noise_compile_dynamic(r)
    test_p6_noise_compile_visibility(r)
    test_a5_1_reserved_word_hints(r)
    test_a5_2_recursive_examples(r)
    test_f5_codegen_scalar_loop_no_crash(r)
    test_f5b_lerp_family_fused_bit_exact(r)
    test_f5c_pow_mod_codegen_fidelity(r)

    # v0.18.0 Phase 1 — precision core
    test_prlp4_fp16_safe_reductions(r)
    test_prlp4_arr_reductions_fp16_safe(r)
    test_prlp2_auto_gate(r)
    test_prlp2_fp16_accuracy_fuzzer(r)
    test_prlp2_node_path_perf(r)
    test_c1_amplification_gate(r)
    test_c2_data_dependent_nan(r)
    test_c2_finiteness_net_recovers(r)
    test_mem2_pool_trim_gating(r)
    test_mem3_fp16_estimator(r)
    test_mem4_per_device_budget(r)

    # v0.18.0 Phase 2 — UX/debugging
    test_dbg1_perf_hud_payload(r)
    test_ux1_diagnostics_reachability(r)
    test_dbg3_nan_overlay(r)
    test_lx5_debug_print(r)
    test_dbg4_doctor(r)
    test_ux2_tooltip_honesty(r)
    test_lx5_json_nan_safe(r)
    test_dbg1_nan_fingerprint(r)

    # v0.18.0 Phase 3 — portability + hardware
    test_port1_import_lint(r)
    test_port1_host_services(r)
    test_port2_facade(r)
    test_port2_program_shape(r)
    test_port3_cli(r)
    test_port3_cli_edges(r)
    test_port3_16bit_png(r)
    test_hw2_multi_gpu_device_context(r)
    test_hw4_cpu_threads(r)

    # v0.18.0 Phase 4 — options & spikes
    test_prlp6_tf32_profile(r)
    test_hw3_triton_validation_skips(r)
    test_hw1_pf1_calibration_smoke(r)
    test_lazy_analysis(r)
    test_lazy_check_status(r)
    test_lazy_execute_path(r)
    test_lazy_schema_pool_ci(r)

    # v0.20.0 Phase 1 — XPU transfer scheduling + tier honesty
    from test_v020_phase1 import (
        test_xpu1_pinned_egress, test_xpu2_unwrap_latent_pinned,
        test_xpu3_nonblocking_ingestion_bitexact, test_xpu4_egress_ingest_roundtrip,
        test_f1_fused_compile_tiers, test_f1b_fused_node_path_reaches_compile_tier,
        test_a2_env_cache_scatter_cow,
        test_c1_gate_profiles_sane, test_g2_verify_arming, test_g1_compile_demotion,
    )
    test_xpu1_pinned_egress(r)
    test_xpu2_unwrap_latent_pinned(r)
    test_xpu3_nonblocking_ingestion_bitexact(r)
    test_xpu4_egress_ingest_roundtrip(r)
    test_f1_fused_compile_tiers(r)
    test_f1b_fused_node_path_reaches_compile_tier(r)
    test_a2_env_cache_scatter_cow(r)
    test_c1_gate_profiles_sane(r)
    test_g2_verify_arming(r)
    test_g1_compile_demotion(r)

    # v0.21.0 Phase 1 — "Fuse the graph" (FUS-1/2/3) + latency/cache/xfer
    from test_v021_phase1 import (
        test_fus1_detector, test_fus1b_multi_injection,
        test_fus3_dag_equivalence, test_fus3_codegen_parity,
        test_fus3_terminal_rmw,
        test_fus1_route_path, test_fus2_fused_lazy, test_fus1_hardening,
        test_cache0_orphan_cg_census, test_lat3_deferred_timing, test_lat4_builtins_lru,
        test_eng8_transfer_model,
    )
    test_fus1_detector(r)
    test_fus1b_multi_injection(r)
    test_fus3_dag_equivalence(r)
    test_fus3_codegen_parity(r)
    test_fus3_terminal_rmw(r)
    test_fus1_route_path(r)
    test_fus2_fused_lazy(r)
    test_fus1_hardening(r)
    test_cache0_orphan_cg_census(r)
    test_lat3_deferred_timing(r)
    test_lat4_builtins_lru(r)
    test_eng8_transfer_model(r)

    # v0.22.0 Phase 1 — "The engine seam" (ENG-1/2/3/4/5/7, SCHED-1, LAT-2)
    from test_v022_phase1 import (
        test_eng3_comfy_profile_canary, test_eng3_engine_profile_preserves_values,
        test_eng1_engine_cooks_without_the_node, test_eng1_node_is_a_marshaller,
        test_eng7_time_builtins_advance, test_eng7_time_barred_from_frozen_tiers,
        test_eng4_structured_compile_error, test_eng5_embedding_canaries,
        test_eng2_null_host_measures_vram, test_eng2_oom_ladder,
        test_eng1_cook_outputs_do_not_alias_inputs, test_eng1_fp16_compiled_tier_clamp,
    )
    test_eng3_comfy_profile_canary(r)
    test_eng3_engine_profile_preserves_values(r)
    test_eng1_engine_cooks_without_the_node(r)
    test_eng1_node_is_a_marshaller(r)
    test_eng7_time_builtins_advance(r)
    test_eng7_time_barred_from_frozen_tiers(r)
    test_eng4_structured_compile_error(r)
    test_eng5_embedding_canaries(r)
    test_eng2_null_host_measures_vram(r)
    test_eng2_oom_ladder(r)
    test_eng1_cook_outputs_do_not_alias_inputs(r)
    test_eng1_fp16_compiled_tier_clamp(r)

    # v0.23.0 Phase 1 — "Authoring" (ROI-1 footprint registry, LANG-1 param metadata)
    from test_v023_phase1 import (
        test_roi1_derivation_matches_historical,
        test_roi1_footprints_wellformed_and_classified,
        test_roi1_malformed_footprint_fails_loud,
        test_roi1_is_tile_safe_unchanged,
        test_lang1_metadata_grammar,
        test_lang1_metadata_literals_only,
        test_lang1_metadata_ignored_by_typecheck,
        test_lang2_check_never_raises,
        test_lang2_w7xxx_warnings,
        test_lang2_sourceloc_end_line,
        test_lang3_version_and_pragma,
        test_lang3_compat_corpus,
        test_eng6_dlpack_contract,
        test_eng9_two_thread_cpu_cook,
        test_lang4_registry_help,
        test_lang5_snippet_store,
        test_lang5_snippet_route,
    )
    test_roi1_derivation_matches_historical(r)
    test_roi1_footprints_wellformed_and_classified(r)
    test_roi1_malformed_footprint_fails_loud(r)
    test_roi1_is_tile_safe_unchanged(r)
    test_lang1_metadata_grammar(r)
    test_lang1_metadata_literals_only(r)
    test_lang1_metadata_ignored_by_typecheck(r)
    test_lang2_check_never_raises(r)
    test_lang2_w7xxx_warnings(r)
    test_lang2_sourceloc_end_line(r)
    test_lang3_version_and_pragma(r)
    test_lang3_compat_corpus(r)
    test_eng6_dlpack_contract(r)
    test_eng9_two_thread_cpu_cook(r)
    test_lang4_registry_help(r)
    test_lang5_snippet_store(r)
    test_lang5_snippet_route(r)

    # v0.24.0 Phase 1 — "See less, cook less" (ROI-2/3/4 spatial laziness, ROI-6 temporal)
    from test_v024_phase1 import (
        test_roi2_footprints,
        test_roi2_plan_executability,
        test_roi4_reach_pinning,
        test_roi4_never_sever,
        test_roi3_tile_is_roi_special_case,
        test_roi4_differential_oracle,
        test_roi4_partition_assembly,
        test_roi4_partial_broadcast_crop,
        test_roi3_engine_integration,
        test_roi6_frame_window,
        test_roi6_batch_strip_equivalence,
        test_roi6_fi_seam_exact,
    )
    test_roi2_footprints(r)
    test_roi2_plan_executability(r)
    test_roi4_reach_pinning(r)
    test_roi4_never_sever(r)
    test_roi3_tile_is_roi_special_case(r)
    test_roi4_differential_oracle(r)
    test_roi4_partition_assembly(r)
    test_roi4_partial_broadcast_crop(r)
    test_roi3_engine_integration(r)
    test_roi6_frame_window(r)
    test_roi6_batch_strip_equivalence(r)
    test_roi6_fi_seam_exact(r)

    # v0.25.0 Phase 1 — "Remember frames" (ENG-12 ownership, CACHE-1 lineage keys,
    # CACHE-2 frame cache, CACHE-3 warm state/prewarm, CACHE-4 layered epochs)
    from test_v025_phase1 import (
        test_eng12_output_is_born_frozen,
        test_eng12_two_strata,
        test_eng12_frozen_frame_reenters_scatter,
        test_cache1_key_construction,
        test_cache1_not_a_content_hash,
        test_cache1_engine_integration,
        test_cache1_playhead_keys,
        test_cache1_precision_and_batch,
        test_cache1_pixel_moving_flags,
        test_cache2_hit_is_bit_exact,
        test_cache2_spill_restore_bit_exact,
        test_cache2_verify_drop_and_replace,
        test_cache3_warm_state_roundtrip,
        test_cache3_version_tag_guard,
        test_cache3_prewarm,
        test_cache4_epoch_tripwire,
        test_cache4_layering,
        test_cache4_codegen_edit_spares_pkl,
        test_cache4_failsafe_oracle,
    )
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

    # v0.26.0 Phase 1 — "Tools" (TOOL-1 manifest/loader/cook, TOOL-3 warm keys, TOOL-4 CLI,
    # TOOL-5 schema+emitter-fuzz, STOCK exemplars, LANG-7 LSP)
    from test_v026_phase1 import (
        test_tool_roundtrip_unfused,
        test_tool_stock_exemplars,
        test_tool_manifest_keys,
        test_tool_promoted_params,
        test_tool_warm_keys,
        test_tool_audit5_fixes,
        test_tool_audit6_fixes,
        test_tool_schema_rejects,
        test_tool_emitter_fuzz,
        test_lsp_smoke,
        test_lsp_bad_frames,
        test_cli_build,
    )
    test_tool_roundtrip_unfused(r)
    test_tool_stock_exemplars(r)
    test_tool_manifest_keys(r)
    test_tool_promoted_params(r)
    test_tool_warm_keys(r)
    test_tool_audit5_fixes(r)
    test_tool_audit6_fixes(r)
    test_tool_schema_rejects(r)
    test_tool_emitter_fuzz(r)
    test_lsp_smoke(r)
    test_lsp_bad_frames(r)
    test_cli_build(r)

    # v0.27.0 Phase 1 — "Big frames, placed well" (SCHED-3 cancel/progress, SCHED-2 placement,
    # CACHE-5 governor, ROI-5 halo tiling, CACHE-6 fusion<->caching)
    from test_v027_phase1 import (
        test_sched3_cancellation,
        test_sched2_placement,
        test_cache5_governor,
        test_roi5_halo_tiling,
        test_cache6_fusion_recook,
    )
    test_sched3_cancellation(r)
    test_sched2_placement(r)
    test_cache5_governor(r)
    test_roi5_halo_tiling(r)
    test_cache6_fusion_recook(r)

    # v0.28.0 Phase 1 — "Second host" (DATA-1 metadata sidecar, DATA-2 storage/EXR/PNG16,
    # DATA-3 array wires, DATA-4 session + soak, PORT-5 the standalone host demo / PM-2)
    from test_v028_phase1 import (
        test_data1_metadata,
        test_data2_storage_exr,
        test_data3_array_wires,
        test_data4_session_soak,
        test_port5_second_host,
        test_data_canaries,
        test_root_channel_and_swizzle_fixes,
    )
    test_data1_metadata(r)
    test_data2_storage_exr(r)
    test_data3_array_wires(r)
    test_data4_session_soak(r)
    test_port5_second_host(r)
    test_data_canaries(r)
    test_root_channel_and_swizzle_fixes(r)

    # v0.29.0 Phase 1 — "Close the register" (ENG-4 re-cut, SCHED-3 bridge, FUS-1b, sweep)
    from test_v029_phase1 import (
        test_eng4_recut_single_raiser,
        test_sched3_bridge_token,
        test_sched3_bridge_node,
        test_count_var_outer_decline,
        test_spatial_scalar_channel_access,
        test_pm5_governor_soak,
    )
    test_eng4_recut_single_raiser(r)
    test_sched3_bridge_token(r)
    test_sched3_bridge_node(r)
    test_count_var_outer_decline(r)
    test_spatial_scalar_channel_access(r)
    test_pm5_governor_soak(r)

    # v0.30 — First viewer
    from test_v030_phase1 import (
        test_v030_roi_host_optin,
        test_v030_roi_malformed_windows,
        test_v030_roi_extent_per_binding,
        test_v030_roi_broadcast_anchor,
        test_v030_roi_refusals_stay_off_the_default_path,
        test_v030_roi_window_is_copied_and_coerced,
        test_v030_roi_never_desyncs_from_its_canvas,
        test_v030_roi_accuracy_envelope,
        test_v030_roi_folded_binding_still_narrows,
        test_v030_comfy_never_arms_roi,
        test_v030_roi_codegen_route_equivalence,
        test_v030_pm6_roi_viewport,
        test_v030_codegen_roi_defaults_off,
        test_v030_nightly_wires_roi_oracle,
    )
    test_v030_roi_host_optin(r)
    test_v030_roi_malformed_windows(r)
    test_v030_roi_extent_per_binding(r)
    test_v030_roi_broadcast_anchor(r)
    test_v030_roi_refusals_stay_off_the_default_path(r)
    test_v030_roi_window_is_copied_and_coerced(r)
    test_v030_roi_never_desyncs_from_its_canvas(r)
    test_v030_roi_accuracy_envelope(r)
    test_v030_roi_folded_binding_still_narrows(r)
    test_v030_comfy_never_arms_roi(r)
    test_v030_roi_codegen_route_equivalence(r)
    test_v030_pm6_roi_viewport(r)
    test_v030_codegen_roi_defaults_off(r)
    test_v030_nightly_wires_roi_oracle(r)

    from test_v031_phase1 import (
        test_v031_sched4_priority_and_preemption,
        test_v031_sched4_finished_work_is_never_discarded,
        test_v031_sched4_outcome_matrix,
        test_v031_sched4_a_foreign_cancel_is_terminal,
        test_v031_sched4_committed_render_completes_under_load,
        test_v031_sched4_worker_survives_a_bad_submit,
        test_v031_sched4_class_contract,
        test_v031_sched4_fifo_and_head_requeue,
        test_v031_sched4_real_cook_preemption,
        test_v031_sched4_off_the_default_path,
    )
    test_v031_sched4_priority_and_preemption(r)
    test_v031_sched4_finished_work_is_never_discarded(r)
    test_v031_sched4_outcome_matrix(r)
    test_v031_sched4_a_foreign_cancel_is_terminal(r)
    test_v031_sched4_committed_render_completes_under_load(r)
    test_v031_sched4_worker_survives_a_bad_submit(r)
    test_v031_sched4_class_contract(r)
    test_v031_sched4_fifo_and_head_requeue(r)
    test_v031_sched4_real_cook_preemption(r)
    test_v031_sched4_off_the_default_path(r)

    from test_v031_phase2 import (
        test_v031_prof1_disarmed_by_default,
        test_v031_prof1_records_and_predicts,
        test_v031_prof1_sampling_gate,
        test_v031_prof1_per_stage_breakdown,
        test_v031_prof1_predicts_an_unseen_resolution,
        test_v031_prof1_ignores_a_failed_cook,
        test_v031_prof1_fused_chains_key_apart,
        test_v031_prof1_state_is_thread_safe,
        test_v031_pred1_bounds_each_factor_not_just_the_product,
        test_v031_pred1_admission_arithmetic,
        test_v031_pred1_unknown_cost_has_a_confidence_brake,
        test_v031_pred1_never_touches_the_other_classes,
        test_v031_pred1_orders_and_sheds_by_value,
        test_v031_pred1_sheds_the_worst_even_after_a_requeue,
        test_v031_pred1_closes_the_loop_with_prof1,
    )
    test_v031_prof1_disarmed_by_default(r)
    test_v031_prof1_records_and_predicts(r)
    test_v031_prof1_sampling_gate(r)
    test_v031_prof1_per_stage_breakdown(r)
    test_v031_prof1_predicts_an_unseen_resolution(r)
    test_v031_prof1_ignores_a_failed_cook(r)
    test_v031_prof1_fused_chains_key_apart(r)
    test_v031_prof1_state_is_thread_safe(r)
    test_v031_pred1_bounds_each_factor_not_just_the_product(r)
    test_v031_pred1_admission_arithmetic(r)
    test_v031_pred1_unknown_cost_has_a_confidence_brake(r)
    test_v031_pred1_never_touches_the_other_classes(r)
    test_v031_pred1_orders_and_sheds_by_value(r)
    test_v031_pred1_sheds_the_worst_even_after_a_requeue(r)
    test_v031_pred1_closes_the_loop_with_prof1(r)

    from test_v031_anim_contract import (
        test_v031_anim_the_spies_are_live,
        test_v031_anim_int_crossing_ramp,
        test_v031_anim_param_sweep_never_recompiles,
        test_v031_anim_negative_control_code_edit_does_recompile,
        test_v031_anim_param_type_matrix,
        test_v031_anim_fused_chain_param_sweep,
        test_v031_anim_textool_promoted_param_sweep,
        test_v031_anim_contract_is_documented,
    )
    test_v031_anim_the_spies_are_live(r)
    test_v031_anim_int_crossing_ramp(r)
    test_v031_anim_param_sweep_never_recompiles(r)
    test_v031_anim_negative_control_code_edit_does_recompile(r)
    test_v031_anim_param_type_matrix(r)
    test_v031_anim_fused_chain_param_sweep(r)
    test_v031_anim_textool_promoted_param_sweep(r)
    test_v031_anim_contract_is_documented(r)

    from test_v031_recovery import (
        test_v031_eng13_atomic_write,
        test_v031_eng13_journal_survives_a_torn_tail,
        test_v031_eng13_warm_state_journals_each_verdict,
        test_v031_eng13_kill_the_process,
        test_v031_eng13_reattach,
    )
    test_v031_eng13_atomic_write(r)
    test_v031_eng13_journal_survives_a_torn_tail(r)
    test_v031_eng13_warm_state_journals_each_verdict(r)
    test_v031_eng13_kill_the_process(r)
    test_v031_eng13_reattach(r)

    # v0.31 NOISE-TIER — the cold frame must render what every later frame renders.
    # (The _TieredCache cold path served an EAGER result and a traced one thereafter;
    #  on CUDA those are not bit-identical, so cook #1 differed from cooks #2+.)
    from test_v031_noise_tiers import (
        test_v031_noise_cold_frame_parity,
        test_v031_noise_resolution_dance,
        test_v031_noise_cold_equals_warm,
        test_v031_noise_stride_signature,
        test_v031_noise_cold_path_shape,
    )
    test_v031_noise_cold_frame_parity(r)
    test_v031_noise_resolution_dance(r)
    test_v031_noise_cold_equals_warm(r)
    test_v031_noise_stride_signature(r)
    test_v031_noise_cold_path_shape(r)

    # v0.31 NOISE-SCALAR — a constant coordinate must render the same on every device.
    # (`fbm(u*8.0, v*8.0, 0.5, 4)` cooked on CPU and raised on CUDA: the GPU-only octave
    #  batching stacked the 0-dim z to [N] and right-aligned it into a spatial axis.)
    from test_v031_noise_scalar_coords import (
        test_v031_noise_scalar_coord_device_parity,
        test_v031_noise_scalar_coord_equals_grid,
        test_v031_noise_scalar_coord_cooks_in_any_slot,
        test_v031_noise_scalar_coord_2d_family,
        test_v031_noise_scalar_coord_reduced_precision,
        test_v031_noise_scalar_coord_batched_equals_per_octave,
        test_v031_noise_scalar_coord_shipped_example,
        test_v031_noise_scalar_coord_helpers_are_noops,
    )
    test_v031_noise_scalar_coord_device_parity(r)
    test_v031_noise_scalar_coord_equals_grid(r)
    test_v031_noise_scalar_coord_cooks_in_any_slot(r)
    test_v031_noise_scalar_coord_2d_family(r)
    test_v031_noise_scalar_coord_reduced_precision(r)
    test_v031_noise_scalar_coord_batched_equals_per_octave(r)
    test_v031_noise_scalar_coord_shipped_example(r)
    test_v031_noise_scalar_coord_helpers_are_noops(r)

    # v0.32 CACHE-7 — effort-based checkpoint placement (docs/effort-based-checkpoints.md).
    from test_v032_checkpoint import (
        test_v032_cache7_differential_oracle,
        test_v032_cache7_fp16_gate_is_lifted_and_exact,
        test_v032_cache7_boundary_key_carries_resolution,
        test_v032_cache7_placement_refuses_rather_than_guesses,
        test_v032_cache7_dag_cut_set_is_analysis_only,
        test_v032_cache7_one_cook_harvests_every_boundary,
        test_v032_cache7_serves_the_deepest_cached_checkpoint,
        test_v032_cache7_upstream_source_key_is_mandatory,
        test_v032_cache7_result_cache_is_thread_safe,
        test_v032_cache7_harvest_respects_the_tap_budget,
        test_v032_cache7_prefix_fingerprint_range_guard,
        test_v032_cache7_suffix_preserves_stage_keys,
        test_v032_cache7_precision_must_be_resolved,
        test_v032_cache7_stage_costs_cross_bucket_fallback,
        test_v032_cache7_off_the_default_path,
        test_v032_cache7_refuses_dag_stage_lists,
        test_v032_cache7_profile_costs_and_confidence_agree,
    )
    test_v032_cache7_differential_oracle(r)
    test_v032_cache7_fp16_gate_is_lifted_and_exact(r)
    test_v032_cache7_boundary_key_carries_resolution(r)
    test_v032_cache7_placement_refuses_rather_than_guesses(r)
    test_v032_cache7_dag_cut_set_is_analysis_only(r)
    test_v032_cache7_one_cook_harvests_every_boundary(r)
    test_v032_cache7_serves_the_deepest_cached_checkpoint(r)
    test_v032_cache7_upstream_source_key_is_mandatory(r)
    test_v032_cache7_result_cache_is_thread_safe(r)
    test_v032_cache7_harvest_respects_the_tap_budget(r)
    test_v032_cache7_prefix_fingerprint_range_guard(r)
    test_v032_cache7_suffix_preserves_stage_keys(r)
    test_v032_cache7_precision_must_be_resolved(r)
    test_v032_cache7_stage_costs_cross_bucket_fallback(r)
    test_v032_cache7_off_the_default_path(r)
    test_v032_cache7_refuses_dag_stage_lists(r)
    test_v032_cache7_profile_costs_and_confidence_agree(r)

    # v0.32 CACHE-9 — region-granular recook (docs/region-granular-recook.md).
    from test_v032_region import (
        test_v032_cache9_region_recook_oracle,
        test_v032_cache9_stale_ring_regression,
        test_v032_cache9_unbounded_reach_inverts_to_whole_frame,
        test_v032_cache9_dirty_from_leaves_clean_stages_alone,
        test_v032_cache9_patch_never_touches_the_cached_master,
        test_v032_cache9_patch_refuses_a_mismatched_window,
        test_v032_cache9_provenance_is_in_the_key,
        test_v032_cache9_second_deeper_edit_needs_valid_regions,
        test_v032_cache9_patch_region_is_atomic,
        test_v032_cache9_second_deeper_edit_pixels,
    )
    test_v032_cache9_region_recook_oracle(r)
    test_v032_cache9_stale_ring_regression(r)
    test_v032_cache9_unbounded_reach_inverts_to_whole_frame(r)
    test_v032_cache9_dirty_from_leaves_clean_stages_alone(r)
    test_v032_cache9_patch_never_touches_the_cached_master(r)
    test_v032_cache9_patch_refuses_a_mismatched_window(r)
    test_v032_cache9_provenance_is_in_the_key(r)
    test_v032_cache9_second_deeper_edit_needs_valid_regions(r)
    test_v032_cache9_patch_region_is_atomic(r)
    test_v032_cache9_second_deeper_edit_pixels(r)

    # v0.32 GOV-1 — memory/effort profiles on the CACHE-5 governor (S item: designs in the
    # CHANGELOG entry, per roadmap §10.1).
    from test_v032_governor import (
        test_v032_gov1_profile_table,
        test_v032_gov1_reaches_the_frame_cache_both_orders,
        test_v032_gov1_tightening_evicts_now,
        test_v032_gov1_owns_the_checkpoint_threshold,
        test_v032_gov1_is_named_and_reportable,
        test_v032_gov1_balanced_restores_the_shipped_budget,
        test_v032_gov1_governed_bytes_never_drifts,
        test_v032_gov1_arbitrate_lands_on_budget_not_on_the_floor,
    )
    test_v032_gov1_profile_table(r)
    test_v032_gov1_reaches_the_frame_cache_both_orders(r)
    test_v032_gov1_tightening_evicts_now(r)
    test_v032_gov1_owns_the_checkpoint_threshold(r)
    test_v032_gov1_is_named_and_reportable(r)
    test_v032_gov1_balanced_restores_the_shipped_budget(r)
    test_v032_gov1_governed_bytes_never_drifts(r)
    test_v032_gov1_arbitrate_lands_on_budget_not_on_the_floor(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
