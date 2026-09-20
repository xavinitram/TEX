#!/usr/bin/env python
"""Standalone TEX test runner — python tests/run_all.py"""
# CACHE-0: point the disk cache at a scratch dir BEFORE any TEX import — get_cache()
# resolves the location once, on first call — so a test run never writes compiled
# artifacts into the shipping package's .tex_cache. setdefault: an outer harness that
# already chose a dir wins.
import os as _os
import sys as _sys
import tempfile as _tempfile
_os.environ.setdefault(
    "TEX_CACHE_DIR", _os.path.join(_tempfile.gettempdir(), "tex_test_cache"))

# B7: a redirected stdout on Windows defaults to cp1252, and the ROI-4 banner in
# test_v024_phase1 contains U+2261 — so `run_all.py > log.txt` died with a UnicodeEncodeError
# partway through a green suite. Force UTF-8 on our own streams rather than asking every
# caller to remember PYTHONIOENCODING (and rather than removing the glyph, which would only
# move the trap to the next one someone types).
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from helpers import SubTestResult

from test_lexer import test_lexer, test_lexer_locations, test_lexer_v11
from test_parser import (
    test_parser, test_parser_v11, test_parser_lvalue_clone,
    test_array_decl_no_hang,
)
from test_type_checker import test_type_checker, test_stdlib_promote_typing, test_select_type_checking
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
    test_ask4_img_size_tier_story,
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
from test_no_unportable_equality_ban import (
    test_no_unportable_equality_ban, test_lnt2_catches_the_three_historical_instances,
)
from test_release_gate import test_version_consistency, test_codegen_determinism, test_scatter_determinism_band
from test_v017_phase1 import (
    test_tst5_tier_trace,
    test_tst6_registry_parity,
    test_tst2_edge_matrix,
    test_tst4_operator_completeness,
    test_tst1_differential_fuzzer, test_a1_1_auto_precision_fuzz,
    test_ask1_convolve_fuzzer_scope,
    test_ask13_patch_dist_fuzzer_scope,
    test_ask4_img_size_fuzzer_scope,
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
    test_dbg4_capabilities_shape, test_dbg4_capabilities_read_only,
    test_dbg4_capabilities_static_truth, test_dbg4_capabilities_measured_truth,
    test_dbg4_capabilities_probe_never_raises, test_dbg4_capabilities_not_folded,
    test_dbg4_doctor_cli_subcommand,
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
    test_select_type_checking(r)
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
    test_ask4_img_size_tier_story(r)
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
    # The opt-in routes learn to place $params on the cook device, so codegen serves
    # param-reading programs on CUDA too; programs already served never place, and the default
    # stencil route is pinned unmoved.
    from test_codegen_param_device import (
        test_codegen_param_served_on_cook_device,
        test_codegen_param_parity_on_every_device,
        test_codegen_param_precision_requests,
        test_codegen_param_placement_learned_once,
        test_codegen_param_default_route_unmoved,
    )
    test_codegen_param_served_on_cook_device(r)
    test_codegen_param_parity_on_every_device(r)
    test_codegen_param_precision_requests(r)
    test_codegen_param_placement_learned_once(r)
    test_codegen_param_default_route_unmoved(r)
    # Native break/continue stays inside the loop that licensed it: a general for-loop's body
    # continues by signal, so the update and the iteration counter below it still run. Each row
    # runs its program in a child process with a timeout, because the defect it pins is a hang.
    from test_codegen_flow_scope import (
        test_flow_scope_emission,
        test_flow_scope_t15_nested_general_loop_terminates,
        test_flow_scope_nested_loop_matrix,
        test_flow_scope_deeper_and_per_pixel_shapes,
    )
    test_flow_scope_emission(r)
    test_flow_scope_t15_nested_general_loop_terminates(r)
    test_flow_scope_nested_loop_matrix(r)
    test_flow_scope_deeper_and_per_pixel_shapes(r)
    # The two values codegen read at the wrong RANK — a vec/colour $param component and
    # `fi` inside a loop — which made per-pixel control flow act uniformly on that tier.
    from test_codegen_value_parity import (
        test_codegen_vec_param_component_matches_interpreter,
        test_codegen_fi_in_loop_matches_interpreter,
        test_codegen_value_parity_on_every_device_and_precision,
        test_codegen_spatial_builtins_match_interpreter_ranks,
        test_codegen_vec_param_staging_is_narrow,
        test_codegen_vec_param_staging_leaves_emitted_code_alone,
    )
    test_codegen_vec_param_component_matches_interpreter(r)
    test_codegen_fi_in_loop_matches_interpreter(r)
    test_codegen_value_parity_on_every_device_and_precision(r)
    test_codegen_spatial_builtins_match_interpreter_ranks(r)
    test_codegen_vec_param_staging_is_narrow(r)
    test_codegen_vec_param_staging_leaves_emitted_code_alone(r)
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
    test_no_unportable_equality_ban(r)
    test_lnt2_catches_the_three_historical_instances(r)
    test_version_consistency(r)
    test_codegen_determinism(r)
    test_tst5_tier_trace(r)
    test_tst6_registry_parity(r)
    test_tst2_edge_matrix(r)
    test_tst4_operator_completeness(r)
    test_tst1_differential_fuzzer(r)
    test_a1_1_auto_precision_fuzz(r)
    test_ask1_convolve_fuzzer_scope(r)
    test_ask13_patch_dist_fuzzer_scope(r)
    test_ask4_img_size_fuzzer_scope(r)
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

    # BRIEF-4 — tex_doctor.capabilities(): a per-tier capability report (C1-C5, C7-C8;
    # C6 is beside the noise-tier promotion tests below).
    test_dbg4_capabilities_shape(r)
    test_dbg4_capabilities_read_only(r)
    test_dbg4_capabilities_static_truth(r)
    test_dbg4_capabilities_measured_truth(r)
    test_dbg4_capabilities_probe_never_raises(r)
    test_dbg4_capabilities_not_folded(r)
    test_dbg4_doctor_cli_subcommand(r)

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
        test_ask4_img_size_fusion,
        test_fus1_route_path, test_fus2_fused_lazy, test_fus1_hardening,
        test_cache0_orphan_cg_census, test_lat3_deferred_timing, test_lat4_builtins_lru,
        test_eng8_transfer_model,
    )
    test_fus1_detector(r)
    test_fus1b_multi_injection(r)
    test_fus3_dag_equivalence(r)
    test_fus3_codegen_parity(r)
    test_fus3_terminal_rmw(r)
    test_ask4_img_size_fusion(r)
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
        test_ask1_convolve_roi_pin,
        test_ask13_patch_dist_roi_pin,
        test_ask4_img_size_roi_pin,
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
    test_ask1_convolve_roi_pin(r)
    test_ask13_patch_dist_roi_pin(r)
    test_ask4_img_size_roi_pin(r)
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
        test_tool_metadata_tooltip_options,
        test_tool_input_optional,
        test_tool_promoted_params,
        test_tool_warm_keys,
        test_tool_audit5_fixes,
        test_tool_audit6_fixes,
        test_tool_schema_rejects,
        test_tool_manifest_byte_identity,
        test_tool_js_publish_filter_pin,
        test_tool_fused_feeds_roundtrip_unfused,
        test_tool_fused_feeds_codegen_parity,
        test_tool_fused_feeds_extent_refusal,
        test_tool_fused_feeds_rejects,
        test_tool_fused_input_refusals_unchanged,
        test_tool_fused_feeds_manifest_roundtrip,
        test_tool_fused_feeds_rekey,
        test_tool_no_feeds_is_pre_feeds_identical,
        test_tool_emitter_fuzz,
        test_lsp_smoke,
        test_lsp_bad_frames,
        test_cli_build,
    )
    test_tool_roundtrip_unfused(r)
    test_tool_stock_exemplars(r)
    test_tool_manifest_keys(r)
    test_tool_metadata_tooltip_options(r)
    test_tool_input_optional(r)
    test_tool_promoted_params(r)
    test_tool_warm_keys(r)
    test_tool_audit5_fixes(r)
    test_tool_audit6_fixes(r)
    test_tool_schema_rejects(r)
    test_tool_manifest_byte_identity(r)
    test_tool_js_publish_filter_pin(r)
    test_tool_fused_feeds_roundtrip_unfused(r)
    test_tool_fused_feeds_codegen_parity(r)
    test_tool_fused_feeds_extent_refusal(r)
    test_tool_fused_feeds_rejects(r)
    test_tool_fused_input_refusals_unchanged(r)
    test_tool_fused_feeds_manifest_roundtrip(r)
    test_tool_fused_feeds_rekey(r)
    test_tool_no_feeds_is_pre_feeds_identical(r)
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
        test_data4_embedding_bringup_runs_as_documented,
        test_data_canaries,
        test_root_channel_and_swizzle_fixes,
    )
    test_data1_metadata(r)
    test_data2_storage_exr(r)
    test_data3_array_wires(r)
    test_data4_session_soak(r)
    test_port5_second_host(r)
    test_data4_embedding_bringup_runs_as_documented(r)
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
    # Exact on each tier; the jit.trace -> Inductor promotion is held to a recorded
    # per-builtin envelope, with its mutation.
    from test_v031_noise_tiers import (
        test_v031_noise_cold_frame_parity,
        test_v031_noise_resolution_dance,
        test_v031_noise_cold_equals_warm,
        test_v031_noise_stride_signature,
        test_v031_noise_cold_path_shape,
        test_v031_noise_promotion_envelope,
        test_v031_noise_promotion_failure_recorded,
        test_v031_noise_tier_record_across_the_promotion,
        test_v031_noise_tier_record_forced,
        test_v031_noise_tier_record_default_path,
    )
    test_v031_noise_cold_frame_parity(r)
    test_v031_noise_resolution_dance(r)
    test_v031_noise_cold_equals_warm(r)
    test_v031_noise_stride_signature(r)
    test_v031_noise_cold_path_shape(r)
    test_v031_noise_promotion_envelope(r)
    test_v031_noise_promotion_failure_recorded(r)  # BRIEF-4 C6
    # The per-cook tier record a host reads before compositing across the promotion
    # (want_noise_tiers -> CookResult.noise_tiers).
    test_v031_noise_tier_record_across_the_promotion(r)
    test_v031_noise_tier_record_forced(r)
    test_v031_noise_tier_record_default_path(r)

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
        test_v032_cache9_a_declined_stage_poisons_a_later_edit,
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
    test_v032_cache9_a_declined_stage_poisons_a_later_edit(r)

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

    # v0.33 PREC-1 — preview-tier storage precision (the deferred decision, argued and pinned).
    from test_v033_precision import (
        test_v033_prec1_default_put_is_unchanged,
        test_v033_prec1_final_tier_is_never_reduced,
        test_v033_prec1_preview_and_final_are_different_keys,
        test_v033_prec1_preview_halves_the_bytes,
        test_v033_prec1_storage_is_invisible_through_get,
        test_v033_prec1_relative_error_is_the_mantissa_bound,
        test_v033_prec1_absolute_error_vs_the_8bit_quantum,
        test_v033_prec1_declines_what_half_cannot_represent,
        test_v033_prec1_survives_the_disk_spill_tier,
        test_v033_prec1_patch_region_inherits_the_base_tier,
        test_v033_prec1_is_absent_from_the_default_comfyui_path,
        test_v033_prec1_choose_storage_is_the_only_decision_point,
        test_v033_prec1_colour_data_split_at_the_kind_seam,
        test_v033_prec1_kind_reaches_the_cache,
        test_v033_prec1_mask_eligible_knob_matrix,
        test_v033_prec1_mask_eligible_packs_through_put,
        test_v033_prec1_mask_eligible_guard_default_refuses,
        test_v033_prec1_mask_eligible_patch_region_ratchet,
        test_v033_prec1_mask_eligible_survives_the_disk_spill_tier,
        test_v033_prec1_preview_is_viral,
    )
    test_v033_prec1_default_put_is_unchanged(r)
    test_v033_prec1_final_tier_is_never_reduced(r)
    test_v033_prec1_preview_and_final_are_different_keys(r)
    test_v033_prec1_preview_halves_the_bytes(r)
    test_v033_prec1_storage_is_invisible_through_get(r)
    test_v033_prec1_relative_error_is_the_mantissa_bound(r)
    test_v033_prec1_absolute_error_vs_the_8bit_quantum(r)
    test_v033_prec1_declines_what_half_cannot_represent(r)
    test_v033_prec1_survives_the_disk_spill_tier(r)
    test_v033_prec1_patch_region_inherits_the_base_tier(r)
    test_v033_prec1_is_absent_from_the_default_comfyui_path(r)
    test_v033_prec1_choose_storage_is_the_only_decision_point(r)
    test_v033_prec1_colour_data_split_at_the_kind_seam(r)
    test_v033_prec1_kind_reaches_the_cache(r)
    test_v033_prec1_mask_eligible_knob_matrix(r)
    test_v033_prec1_mask_eligible_packs_through_put(r)
    test_v033_prec1_mask_eligible_guard_default_refuses(r)
    test_v033_prec1_mask_eligible_patch_region_ratchet(r)
    test_v033_prec1_mask_eligible_survives_the_disk_spill_tier(r)
    test_v033_prec1_preview_is_viral(r)

    # v0.33 CACHE-8 — deep cache tiers: the residency ladder, the packing choices, and the
    # entropy codec the measured Pareto rejected (docs/compressed-cache-tiers.md).
    from test_v033_cache8 import (
        test_v033_cache8_residency_is_off_until_armed,
        test_v033_cache8_demote_frees_vram_and_keeps_the_frame,
        test_v033_cache8_demoted_frame_is_bit_exact,
        test_v033_cache8_promote_on_reuse_returns_it_home,
        test_v033_cache8_a_spilled_demoted_frame_comes_back_to_its_home,
        test_v033_cache8_governor_prefers_demotion_over_eviction,
        test_v033_cache8_unarmed_governor_evicts_exactly_as_before,
        test_v033_cache8_uint16_is_offered_never_chosen,
        test_v033_cache8_uint16_beats_fp16_in_range,
        test_v033_cache8_no_compression_path_is_switched_on,
        test_v033_cache8_profiles_carry_the_residency_ceiling,
        test_v033_cache8_profile_reaches_and_restores_the_ceiling,
        test_v033_cache8_is_absent_from_the_default_comfyui_path,
        test_v033_cache8_touch_is_not_a_read,
        test_v033_cache8_touch_steers_the_victim_walks,
        test_v033_cache8_touch_leaves_the_demanded_frame_on_top,
        test_v033_cache8_touch_never_moves_a_frame_between_devices,
        test_v033_cache8_a_touched_frame_serves_bit_exact,
        test_v033_cache8_membership_is_a_pure_question,
        test_v033_cache8_touch_changes_no_pool_share_under_the_governor,
        test_v033_cache8_touch_and_in_survive_a_threaded_race,
        test_v033_cache8_touch_is_atomic_against_a_concurrent_put,
        test_v033_cache8_hints_never_wait_on_or_undo_in_flight_work,
    )
    test_v033_cache8_residency_is_off_until_armed(r)
    test_v033_cache8_demote_frees_vram_and_keeps_the_frame(r)
    test_v033_cache8_demoted_frame_is_bit_exact(r)
    test_v033_cache8_promote_on_reuse_returns_it_home(r)
    test_v033_cache8_a_spilled_demoted_frame_comes_back_to_its_home(r)
    test_v033_cache8_governor_prefers_demotion_over_eviction(r)
    test_v033_cache8_unarmed_governor_evicts_exactly_as_before(r)
    test_v033_cache8_uint16_is_offered_never_chosen(r)
    test_v033_cache8_uint16_beats_fp16_in_range(r)
    test_v033_cache8_no_compression_path_is_switched_on(r)
    test_v033_cache8_profiles_carry_the_residency_ceiling(r)
    test_v033_cache8_profile_reaches_and_restores_the_ceiling(r)
    test_v033_cache8_is_absent_from_the_default_comfyui_path(r)
    # Residency hints: `touch` / `in` steer the victim walks without reading.
    test_v033_cache8_touch_is_not_a_read(r)
    test_v033_cache8_touch_steers_the_victim_walks(r)
    test_v033_cache8_touch_leaves_the_demanded_frame_on_top(r)
    test_v033_cache8_touch_never_moves_a_frame_between_devices(r)
    test_v033_cache8_a_touched_frame_serves_bit_exact(r)
    test_v033_cache8_membership_is_a_pure_question(r)
    test_v033_cache8_touch_changes_no_pool_share_under_the_governor(r)
    test_v033_cache8_touch_and_in_survive_a_threaded_race(r)
    test_v033_cache8_touch_is_atomic_against_a_concurrent_put(r)
    test_v033_cache8_hints_never_wait_on_or_undo_in_flight_work(r)

    # v0.33 XPU-2 — engine-owned async D2H egress. The fence row is the release exit gate
    # ("egress fences proven by a stress test that consumes frames from the wrong side").
    from test_v033_xpu2 import (
        test_v033_xpu2_handle_metadata_never_fences,
        test_v033_xpu2_fenced_read_is_bit_exact,
        test_v033_xpu2_the_fence_is_load_bearing,
        test_v033_xpu2_declines_asynchrony_rather_than_faking_it,
        test_v033_xpu2_wait_is_idempotent_and_releases_the_source,
        test_v033_xpu2_spill_round_trips_through_the_handle,
        test_v033_xpu2_is_engine_only,
    )
    test_v033_xpu2_handle_metadata_never_fences(r)
    test_v033_xpu2_fenced_read_is_bit_exact(r)
    test_v033_xpu2_the_fence_is_load_bearing(r)
    test_v033_xpu2_declines_asynchrony_rather_than_faking_it(r)
    test_v033_xpu2_wait_is_idempotent_and_releases_the_source(r)
    test_v033_xpu2_spill_round_trips_through_the_handle(r)
    test_v033_xpu2_is_engine_only(r)

    # v0.33 Phase 0 — the fix-first register from the v0.30-v0.32 audit. Every row fails on the
    # pre-fix tree; P0-2's counterexample lives in test_v032_checkpoint (the inverted row).
    from test_v033_phase0 import (
        test_v033_p0_3_generator_head_key_carries_resolution,
        test_v033_p0_4_chain_windows_guards_a_past_the_end_start,
        test_v033_p0_4_a_decline_poisons_validity,
        test_v033_p0_6_reindex_does_not_lose_a_racing_spill,
        test_v033_p0_6_spill_index_mutations_are_locked,
        test_v033_p0_7_mkstemp_retry_is_bounded,
        test_v033_p0_7_every_mkstemp_site_is_bounded,
        test_v033_p0_8_empty_cache_short_circuits,
        test_v033_p0_8_default_budget_probe_is_memoized,
        test_v033_p0_8_docstring_no_longer_claims_free,
        test_v033_p0_5_tap_keys_survive_every_cook_path,
    )
    test_v033_p0_3_generator_head_key_carries_resolution(r)
    test_v033_p0_4_chain_windows_guards_a_past_the_end_start(r)
    test_v033_p0_4_a_decline_poisons_validity(r)
    test_v033_p0_6_reindex_does_not_lose_a_racing_spill(r)
    test_v033_p0_6_spill_index_mutations_are_locked(r)
    test_v033_p0_7_mkstemp_retry_is_bounded(r)
    test_v033_p0_7_every_mkstemp_site_is_bounded(r)
    test_v033_p0_8_empty_cache_short_circuits(r)
    test_v033_p0_8_default_budget_probe_is_memoized(r)
    test_v033_p0_8_docstring_no_longer_claims_free(r)
    test_v033_p0_5_tap_keys_survive_every_cook_path(r)

    # v0.33.1 — the v0.33.0 release-audit findings. Every row fails on the pre-fix
    # tree (verified against a `git archive HEAD` checkout, 10/10).
    from test_v0331_audit import (
        test_v0331_a1_double_demotion_cannot_skew_the_byte_totals,
        test_v0331_a1_a_demoting_frame_is_not_requeued,
        test_v0331_a2_restore_returns_the_representation_atomically,
        test_v0331_a2_a_racing_clear_never_serves_storage_dtype,
        test_v0331_a3_reindex_never_rebinds_over_a_racing_spill,
        test_v0331_a4_every_storage_dtype_survives_the_spill_tier,
        test_v0331_a4_frame_records_carry_a_format_version,
        test_v0331_a5_clear_is_not_undone_by_an_inflight_spill,
        test_v0331_a7_the_mru_frame_is_never_demoted,
        test_v0331_a8_only_fp32_sources_are_packed,
        test_v0331_a1_a_duplicate_queue_entry_commits_once,
        test_v0331_a3_a_learned_membership_set_also_survives_the_scan,
    )
    test_v0331_a1_double_demotion_cannot_skew_the_byte_totals(r)
    test_v0331_a1_a_demoting_frame_is_not_requeued(r)
    test_v0331_a2_restore_returns_the_representation_atomically(r)
    test_v0331_a2_a_racing_clear_never_serves_storage_dtype(r)
    test_v0331_a3_reindex_never_rebinds_over_a_racing_spill(r)
    test_v0331_a4_every_storage_dtype_survives_the_spill_tier(r)
    test_v0331_a4_frame_records_carry_a_format_version(r)
    test_v0331_a5_clear_is_not_undone_by_an_inflight_spill(r)
    test_v0331_a7_the_mru_frame_is_never_demoted(r)
    test_v0331_a8_only_fp32_sources_are_packed(r)
    test_v0331_a1_a_duplicate_queue_entry_commits_once(r)
    test_v0331_a3_a_learned_membership_set_also_survives_the_scan(r)

    # v0.33.2 — the v0.33.1 release-audit findings. Every row fails on the pre-fix
    # tree (verified against a `git archive v0.33.1` checkout, 11/11).
    from test_v0332_audit import (
        test_v0332_a1_a_stale_spill_never_overwrites_the_winner,
        test_v0332_a1_a_check_and_write_are_one_critical_section,
        test_v0332_a2_a_restore_cannot_resurrect_a_cleared_frame,
        test_v0332_a3_clear_does_not_orphan_a_frame_spilled_during_its_walk,
        test_v0332_a4_the_stored_quality_tag_survives_the_disk_tier,
        test_v0332_a4_patch_region_cannot_launder_a_preview_base,
        test_v0332_a4_the_viral_rule_reaches_the_disk_tier_and_spares_bare_bases,
        test_v0332_a5_propagate_quality_accepts_a_single_tag,
        test_v0332_a5_a_future_frame_format_is_refused,
        test_v0332_a5_patch_region_refuses_a_mismatch_instead_of_raising,
        test_v0332_a5_promote_keeps_a_patched_frame_on_its_home_device,
        test_v0332_a5_disarming_residency_cancels_queued_demotions,
        test_v0332_a5_disarming_also_cancels_a_demotion_already_in_flight,
        test_v0332_h1_the_spill_ticket_orders_by_put_not_by_drain_start,
        test_v0332_h2_a_restore_cannot_outrun_clears_unlink_walk,
        test_v0332_h3_the_ratchet_never_packs_a_frame_put_refused_to_pack,
        test_v0332_h4_learn_spilled_does_not_orphan_a_racing_spill,
        test_v0332_h5_a_failed_spill_write_is_not_counted_or_indexed,
        test_v0332_h6_a_disarmed_cache_issues_no_demotion_copy,
        test_v0332_h7_a_restore_that_starts_inside_a_purge_is_refused_at_capture,
        test_v0332_h7_the_purge_depth_survives_an_interrupted_walk,
    )
    test_v0332_a1_a_stale_spill_never_overwrites_the_winner(r)
    test_v0332_a1_a_check_and_write_are_one_critical_section(r)
    test_v0332_a2_a_restore_cannot_resurrect_a_cleared_frame(r)
    test_v0332_a3_clear_does_not_orphan_a_frame_spilled_during_its_walk(r)
    test_v0332_a4_the_stored_quality_tag_survives_the_disk_tier(r)
    test_v0332_a4_patch_region_cannot_launder_a_preview_base(r)
    test_v0332_a4_the_viral_rule_reaches_the_disk_tier_and_spares_bare_bases(r)
    test_v0332_a5_propagate_quality_accepts_a_single_tag(r)
    test_v0332_a5_a_future_frame_format_is_refused(r)
    test_v0332_a5_patch_region_refuses_a_mismatch_instead_of_raising(r)
    test_v0332_a5_promote_keeps_a_patched_frame_on_its_home_device(r)
    test_v0332_a5_disarming_residency_cancels_queued_demotions(r)
    test_v0332_a5_disarming_also_cancels_a_demotion_already_in_flight(r)
    test_v0332_h1_the_spill_ticket_orders_by_put_not_by_drain_start(r)
    test_v0332_h2_a_restore_cannot_outrun_clears_unlink_walk(r)
    test_v0332_h3_the_ratchet_never_packs_a_frame_put_refused_to_pack(r)
    test_v0332_h4_learn_spilled_does_not_orphan_a_racing_spill(r)
    test_v0332_h5_a_failed_spill_write_is_not_counted_or_indexed(r)
    test_v0332_h6_a_disarmed_cache_issues_no_demotion_copy(r)
    test_v0332_h7_a_restore_that_starts_inside_a_purge_is_refused_at_capture(r)
    test_v0332_h7_the_purge_depth_survives_an_interrupted_walk(r)

    # v0.34 DATA-7 — the host source protocol (fetch_time/sample_time) and its pool.
    from test_v034_data7 import (
        test_v034_data7_null_provider_refuses,
        test_v034_data7_reads_cook_on_both_devices,
        test_v034_data7_motion_blur_exemplar,
        test_v034_data7_temporal_median_exemplar,
        test_v034_data7_a_per_pixel_time_is_refused,
        test_v034_data7_a_bad_frame_is_refused,
        test_v034_data7_a_provider_failure_is_named,
        test_v034_data7_quantization_collapses_neighbouring_times,
        test_v034_data7_no_source_key_never_caches,
        test_v034_data7_a_version_bump_invalidates,
        test_v034_data7_refuses_tiling_and_graph_capture,
        test_v034_data7_unarmed_costs_nothing,
        test_v034_data7_pool_arbitrates_under_the_governor,
        test_v034_data7_eviction_is_playhead_aware,
        test_v034_data7_reattach_reports_the_media_pool,
        test_v034_data7_speculative_io_failure_never_alarms,
    )
    test_v034_data7_null_provider_refuses(r)
    test_v034_data7_reads_cook_on_both_devices(r)
    test_v034_data7_motion_blur_exemplar(r)
    test_v034_data7_temporal_median_exemplar(r)
    test_v034_data7_a_per_pixel_time_is_refused(r)
    test_v034_data7_a_bad_frame_is_refused(r)
    test_v034_data7_a_provider_failure_is_named(r)
    test_v034_data7_quantization_collapses_neighbouring_times(r)
    test_v034_data7_no_source_key_never_caches(r)
    test_v034_data7_a_version_bump_invalidates(r)
    test_v034_data7_refuses_tiling_and_graph_capture(r)
    test_v034_data7_unarmed_costs_nothing(r)
    test_v034_data7_pool_arbitrates_under_the_governor(r)
    test_v034_data7_eviction_is_playhead_aware(r)
    test_v034_data7_reattach_reports_the_media_pool(r)
    test_v034_data7_speculative_io_failure_never_alarms(r)

    # v0.34 IO-1 — promised bindings + dependency-aware admission, and the §3.3
    # async-write contract (a test, not a subsystem).
    from test_v034_io1 import (
        test_v034_io1_unknown_bindings_are_refused,
        test_v034_io1_a_promise_declares_its_identity,
        test_v034_io1_prepare_resolves_and_refuses,
        test_v034_io1_a_ready_branch_cooks_while_its_sibling_waits,
        test_v034_io1_a_waiting_submit_never_preempts,
        test_v034_io1_a_failed_promise_fails_its_jobs,
        test_v034_io1_waiting_jobs_stay_visible_to_shed_and_close,
        test_v034_io1_prefetch_never_feeds_the_profiler,
        test_v034_io1_declare_window_prefetches_a_range,
        test_v034_io1_backpressure_refuses_rather_than_evicting,
        test_v034_io1_cancellation_drops_on_landing,
        test_v034_async_write_does_not_block_the_next_cook,
        test_v034_async_write_bytes_are_bit_exact,
    )
    test_v034_io1_unknown_bindings_are_refused(r)
    test_v034_io1_a_promise_declares_its_identity(r)
    test_v034_io1_prepare_resolves_and_refuses(r)
    test_v034_io1_a_ready_branch_cooks_while_its_sibling_waits(r)
    test_v034_io1_a_waiting_submit_never_preempts(r)
    test_v034_io1_a_failed_promise_fails_its_jobs(r)
    test_v034_io1_waiting_jobs_stay_visible_to_shed_and_close(r)
    test_v034_io1_prefetch_never_feeds_the_profiler(r)
    test_v034_io1_declare_window_prefetches_a_range(r)
    test_v034_io1_backpressure_refuses_rather_than_evicting(r)
    test_v034_io1_cancellation_drops_on_landing(r)
    test_v034_async_write_does_not_block_the_next_cook(r)
    test_v034_async_write_bytes_are_bit_exact(r)

    # v0.34.1 — the v0.34.0 release-audit findings. Every row verified to fail on a
    # pristine worktree at v0.34.0 (9/9 reproduced pre-fix, 0/9 after).
    from test_v0341_audit import (
        test_v0341_a_a_provider_swap_never_serves_the_replaced_pixels,
        test_v0341_b_cancelling_a_waiting_job_terminates_it,
        test_v0341_c_a_provider_frame_is_fp32_or_refused,
        test_v0341_d_constant_coordinates_stay_on_the_cook_grid,
        test_v0341_e_the_pool_owns_the_bytes_it_accounts,
        test_v0341_f_a_failed_promise_never_alarms_speculative_work,
        test_v0341_g_a_promise_cannot_land_none,
        test_v0341_h_the_stage_list_family_understands_promises,
        test_v0341_i_high_rank_tensors_and_helper_locations,
        test_v0341_h_promise_prefixes_key_by_resolution,
        test_v0341_d_the_cook_grid_agrees_across_tiers,
        test_v0341_e_a_buffer_reusing_provider_is_still_copied,
        test_v0341_c_a_half_source_stays_half_in_the_pool,
        test_v0341_g_fail_none_is_refused_too,
    )
    test_v0341_a_a_provider_swap_never_serves_the_replaced_pixels(r)
    test_v0341_b_cancelling_a_waiting_job_terminates_it(r)
    test_v0341_c_a_provider_frame_is_fp32_or_refused(r)
    test_v0341_d_constant_coordinates_stay_on_the_cook_grid(r)
    test_v0341_e_the_pool_owns_the_bytes_it_accounts(r)
    test_v0341_f_a_failed_promise_never_alarms_speculative_work(r)
    test_v0341_g_a_promise_cannot_land_none(r)
    test_v0341_h_the_stage_list_family_understands_promises(r)
    test_v0341_i_high_rank_tensors_and_helper_locations(r)
    test_v0341_h_promise_prefixes_key_by_resolution(r)
    test_v0341_d_the_cook_grid_agrees_across_tiers(r)
    test_v0341_e_a_buffer_reusing_provider_is_still_copied(r)
    test_v0341_c_a_half_source_stays_half_in_the_pool(r)
    test_v0341_g_fail_none_is_refused_too(r)

    # v0.34 R2-archive — the compat-corpus goldens become an append-only per-version
    # archive. Landed a release early, while it is still neutral (one frozen version).
    from test_v034_r2_archive import (
        test_v034_r2_freeze_may_only_add,
        test_v034_r2_versions_sort_numerically,
        test_v034_r2_neutrality,
        test_v034_r2_regen_is_gone,
    )
    test_v034_r2_freeze_may_only_add(r)
    test_v034_r2_versions_sort_numerically(r)
    test_v034_r2_neutrality(r)
    test_v034_r2_regen_is_gone(r)

    # v0.35 CF-7 hygiene: pins for three facts synchronised by comment.
    from test_v035_hygiene import (
        test_v035_cf7_builtin_name_sets_agree,
        test_v035_cf7_js_publish_manifest_tracks_the_language_version,
        test_v035_cf6_the_grid_is_a_consensus_not_first_wins,
        test_v035_cf6_an_unread_binding_does_not_size_the_grid,
        test_v035_cf6_the_roi_grid_uses_the_same_participants,
        test_v035_cf6_the_peak_estimate_describes_the_grid_the_cook_uses,
        test_v035_cf6_binding_order_does_not_move_the_auto_precision_gate,
        test_v035_cf2_a_whole_frame_partial_recook_checks_its_prefix,
        test_v035_cf4_requalify_lands_the_final_and_evicts_the_preview,
        test_v035_cf1_a_patch_over_a_demoted_base_keeps_its_home,
        test_v035_port6_engine_import_is_adapter_free,
        test_v035_port6_routes_still_register_under_comfyui,
    )
    test_v035_cf7_builtin_name_sets_agree(r)
    test_v035_cf7_js_publish_manifest_tracks_the_language_version(r)
    test_v035_cf6_the_grid_is_a_consensus_not_first_wins(r)
    test_v035_cf6_an_unread_binding_does_not_size_the_grid(r)
    test_v035_cf6_the_roi_grid_uses_the_same_participants(r)
    test_v035_cf6_the_peak_estimate_describes_the_grid_the_cook_uses(r)
    test_v035_cf6_binding_order_does_not_move_the_auto_precision_gate(r)
    test_v035_cf2_a_whole_frame_partial_recook_checks_its_prefix(r)
    test_v035_cf4_requalify_lands_the_final_and_evicts_the_preview(r)
    test_v035_cf1_a_patch_over_a_demoted_base_keeps_its_home(r)
    test_v035_port6_engine_import_is_adapter_free(r)
    test_v035_port6_routes_still_register_under_comfyui(r)

    # Uniform outputs (LANGUAGE.md §5.2) — a once-per-cook scalar `@` output, computed only
    # from literals, scalar params and the 0-dim builtins, is pinned identical across the
    # interpreter, codegen, every compile_mode, the tiled/batch-strip/ROI assemblers and a
    # fused chain's terminal stage. Docs + tests only; no engine file changed.
    from test_v035_hygiene import (
        test_brief9_t1_uniform_output_interp_codegen_equal,
        test_brief9_t2_uniform_output_stable_across_compile_modes,
        test_brief9_t3_uniform_output_survives_tiled_batch_roi_assemblers,
        test_brief9_t4_uniform_output_fusion_terminal_and_midchain_refusal,
        test_brief9_t5_uniform_output_param_query_never_recompiles_moves_lineage,
        test_brief9_t6_uniform_output_fp32_exact_at_3841,
        test_brief9_t7_language_md_documents_uniform_outputs,
    )
    test_brief9_t1_uniform_output_interp_codegen_equal(r)
    test_brief9_t2_uniform_output_stable_across_compile_modes(r)
    test_brief9_t3_uniform_output_survives_tiled_batch_roi_assemblers(r)
    test_brief9_t4_uniform_output_fusion_terminal_and_midchain_refusal(r)
    test_brief9_t5_uniform_output_param_query_never_recompiles_moves_lineage(r)
    test_brief9_t6_uniform_output_fp32_exact_at_3841(r)
    test_brief9_t7_language_md_documents_uniform_outputs(r)

    # ASK-1 — native convolve builtin (T8 edge cases, T11 reserved name; the other rows
    # land as edits to the existing test files wired above, e.g. test_v017_phase1.py /
    # test_v024_phase1.py).
    from test_v035_ask1 import (
        test_ask1_t8_1x1_kernel_is_scale,
        test_ask1_t8_2x2_even_kernel_centering,
        test_ask1_t8_nan_kernel_propagates,
        test_ask1_t8_kernel_wider_than_image,
        test_ask1_t8_invalid_args_raise,
        test_ask1_t11_e3011_reserved_name,
    )
    test_ask1_t8_1x1_kernel_is_scale(r)
    test_ask1_t8_2x2_even_kernel_centering(r)
    test_ask1_t8_nan_kernel_propagates(r)
    test_ask1_t8_kernel_wider_than_image(r)
    test_ask1_t8_invalid_args_raise(r)
    test_ask1_t11_e3011_reserved_name(r)

    # ASK-13 — patch-distance builtin (T6 edge cases + the reserved-name row; the other
    # rows land as edits to the existing test files that own each property, wired
    # above/near the ASK-1 block).
    from test_v035_ask13 import (
        test_ask13_t6_radius0_equals_pointwise,
        test_ask13_t6_zero_offset_exact_zero,
        test_ask13_t6_large_offset_finite_clamped,
        test_ask13_t6_nan_propagates,
        test_ask13_t6_per_pixel_offset_raises,
        test_ask13_reserved_name_e3011,
        test_ask13_patch_dist_offset_clamp_pad_bound,
        test_ask13_patch_dist_offset_clamp_bitexact,
    )
    test_ask13_t6_radius0_equals_pointwise(r)
    test_ask13_t6_zero_offset_exact_zero(r)
    test_ask13_t6_large_offset_finite_clamped(r)
    test_ask13_t6_nan_propagates(r)
    test_ask13_t6_per_pixel_offset_raises(r)
    test_ask13_reserved_name_e3011(r)
    test_ask13_patch_dist_offset_clamp_pad_bound(r)
    test_ask13_patch_dist_offset_clamp_bitexact(r)

    # ASK-4 — img_width/img_height, a binding's own width and height (values on both
    # tiers/devices, mask binding, uniform=1, fp16-precision exact reading, graph
    # tier, E3011/E5003; the other rows land as edits to the existing test files
    # wired above, e.g. test_v017_phase2.py / test_v021_phase1.py / test_v024_phase1.py).
    from test_v036_ask4 import (
        test_ask4_t1_width_height_values,
        test_ask4_t2_mask_binding,
        test_ask4_t3_uniform_reads_one,
        test_ask4_t4_fp16_precision_exact_at_4095,
        test_ask4_t5_graph_tier,
        test_ask4_t6_e3011_reserved_names,
        test_ask4_t7_e5003_argument_type,
    )
    test_ask4_t1_width_height_values(r)
    test_ask4_t2_mask_binding(r)
    test_ask4_t3_uniform_reads_one(r)
    test_ask4_t4_fp16_precision_exact_at_4095(r)
    test_ask4_t5_graph_tier(r)
    test_ask4_t6_e3011_reserved_names(r)
    test_ask4_t7_e5003_argument_type(r)

    # ASK-5 — a per-cell worley_id builtin (voronoi's own output stays unchanged; only
    # its help text was wrong). The other rows (taxonomy/fuzzer/edge-matrix/fp16-loud-
    # guard/_NON_LOCAL_FNS) land as coverage the EXISTING suites pick up automatically
    # once worley_id is registered footprint='point' — no edit needed there.
    from test_v036_ask5 import (
        test_ask5_reserved_name_e3011,
        test_ask5_determinism_and_cell_count,
        test_ask5_matches_worley_f1_winner,
        test_ask5_worley_f1_tier_envelope,
        test_ask5_voronoi_unchanged,
        test_ask5_help_text_corrected,
        test_ask5_taxonomy_pin,
        test_ask5_fp16_fragile_gate,
        test_ask5_eager_only_no_promotion,
        test_ask5_both_tiers_bit_exact,
        test_ask5_graph_tier_bit_exact,
    )
    test_ask5_reserved_name_e3011(r)
    test_ask5_determinism_and_cell_count(r)
    test_ask5_matches_worley_f1_winner(r)
    test_ask5_worley_f1_tier_envelope(r)
    test_ask5_voronoi_unchanged(r)
    test_ask5_help_text_corrected(r)
    test_ask5_taxonomy_pin(r)
    test_ask5_fp16_fragile_gate(r)
    test_ask5_eager_only_no_promotion(r)
    test_ask5_both_tiers_bit_exact(r)
    test_ask5_graph_tier_bit_exact(r)

    # HOOK-3: the linear collapse a host needs, and the gate's refusal as data.
    from test_hook3_checkpoint_collapse import (
        test_hook3_collapse_linear_on_a_real_region,
        test_hook3_collapse_refuses_non_linear,
        test_hook3_gate_refusal_is_structured,
        test_hook3_gate_decision_is_unchanged,
    )
    test_hook3_collapse_linear_on_a_real_region(r)
    test_hook3_collapse_refuses_non_linear(r)
    test_hook3_gate_refusal_is_structured(r)
    test_hook3_gate_decision_is_unchanged(r)

    # HOOK-2: a per-document binding map on tex_lsp's diagnostics path (didOpen/didChange).
    from test_hook2_lsp_bindings import (
        test_hook2_binding_map_surfaces_the_vec4_assumption,
        test_hook2_no_map_is_byte_for_byte_the_base_sha,
        test_hook2_didchange_without_a_map_keeps_the_previous_map,
        test_hook2_malformed_map_never_crashes,
    )
    test_hook2_binding_map_surfaces_the_vec4_assumption(r)
    test_hook2_no_map_is_byte_for_byte_the_base_sha(r)
    test_hook2_didchange_without_a_map_keeps_the_previous_map(r)
    test_hook2_malformed_map_never_crashes(r)

    # HOOK-4: the state-isolation kit, reachable as TEX_Wrangle.tex_testkit.
    from test_hook4_testkit import (
        test_hook4_tex_testkit_importable_without_tests_on_syspath,
        test_hook4_bare_star_import_yields_the_base_sha_set,
        test_hook4_helpers_reexports_are_tex_testkit_objects,
        test_hook4_testkit_stays_off_pytest_and_the_comfy_adapter_path,
    )
    test_hook4_tex_testkit_importable_without_tests_on_syspath(r)
    test_hook4_bare_star_import_yields_the_base_sha_set(r)
    test_hook4_helpers_reexports_are_tex_testkit_objects(r)
    test_hook4_testkit_stays_off_pytest_and_the_comfy_adapter_path(r)

    # BRIEF-10: integrity BEFORE deserialise in the on-disk pickle caches.
    from test_brief10_integrity import (
        test_brief10_pkl_site_never_executes_crafted_reduce,
        test_brief10_cg_site_never_executes_crafted_reduce,
        test_brief10_frame_site_never_executes_crafted_reduce,
        test_brief10_pkl_site_rejects_a_forged_trailer,
        test_brief10_cg_site_rejects_a_forged_trailer,
        test_brief10_frame_site_rejects_a_forged_trailer,
        test_brief10_signed_entries_still_load,
        test_brief10_unsigned_pkl_is_a_silent_miss_then_recompiles,
        test_brief10_unsigned_frame_is_a_silent_miss,
        test_brief10_key_repair_spares_a_peers_republished_key,
    )
    test_brief10_pkl_site_never_executes_crafted_reduce(r)
    test_brief10_cg_site_never_executes_crafted_reduce(r)
    test_brief10_frame_site_never_executes_crafted_reduce(r)
    test_brief10_pkl_site_rejects_a_forged_trailer(r)
    test_brief10_cg_site_rejects_a_forged_trailer(r)
    test_brief10_frame_site_rejects_a_forged_trailer(r)
    test_brief10_signed_entries_still_load(r)
    test_brief10_unsigned_pkl_is_a_silent_miss_then_recompiles(r)
    test_brief10_unsigned_frame_is_a_silent_miss(r)
    test_brief10_key_repair_spares_a_peers_republished_key(r)

    # Per-pixel control flow (LANGUAGE.md §7.1): the rule pinned on both tiers, the doc text,
    # and the opt-in W7006/W7007 advisories, which never reach check().
    from test_v0351_control_flow import (
        test_control_flow_per_pixel_condition_semantics_both_tiers,
        test_control_flow_language_md_states_the_rule,
        test_control_flow_w7006_marks_gathers_in_per_pixel_branches,
        test_control_flow_w7007_marks_control_flow_on_every_pixel,
        test_control_flow_advisories_spare_uniform_shapes,
        test_control_flow_advisories_are_invisible_to_check,
        test_control_flow_per_pixel_loop_bound_semantics_both_tiers,
        test_control_flow_language_md_states_the_loop_bound,
        test_control_flow_fix_pixels_matches_a_per_pixel_reference,
        test_control_flow_break_search_matches_a_per_pixel_reference,
        test_control_flow_custom_blend_matches_a_per_pixel_reference,
        test_control_flow_while_loop_matches_a_per_pixel_reference,
        test_control_flow_vector_blur_matches_a_per_pixel_reference,
        test_control_flow_recursive_pattern_matches_a_per_pixel_reference,
        test_control_flow_fixed_examples_keep_their_surface,
    )
    test_control_flow_per_pixel_condition_semantics_both_tiers(r)
    test_control_flow_language_md_states_the_rule(r)
    test_control_flow_w7006_marks_gathers_in_per_pixel_branches(r)
    test_control_flow_w7007_marks_control_flow_on_every_pixel(r)
    test_control_flow_advisories_spare_uniform_shapes(r)
    test_control_flow_advisories_are_invisible_to_check(r)
    test_control_flow_per_pixel_loop_bound_semantics_both_tiers(r)
    test_control_flow_language_md_states_the_loop_bound(r)
    test_control_flow_fix_pixels_matches_a_per_pixel_reference(r)
    test_control_flow_break_search_matches_a_per_pixel_reference(r)
    test_control_flow_custom_blend_matches_a_per_pixel_reference(r)
    test_control_flow_while_loop_matches_a_per_pixel_reference(r)
    test_control_flow_vector_blur_matches_a_per_pixel_reference(r)
    test_control_flow_recursive_pattern_matches_a_per_pixel_reference(r)
    test_control_flow_fixed_examples_keep_their_surface(r)

    # TRK-24: a negative literal `$param` default (`f$k=-0.3;`) was silently dropped to
    # None by the type checker's default-extraction chain; the JS widget builder already
    # read it from source text independently, so only bare cook()/CLI/API callers saw it.
    from test_trk24_negative_param_defaults import (
        test_negative_literal_defaults_recorded,
        test_positive_defaults_unchanged,
        test_unary_op_ast_shape_matches_assumption,
        test_shipped_examples_declared_defaults,
        test_bare_cook_omitted_negative_param_uses_default,
        test_bare_cook_omitted_param_matches_explicit_default,
        test_tex_node_widget_defaults_unchanged,
    )
    test_negative_literal_defaults_recorded(r)
    test_positive_defaults_unchanged(r)
    test_unary_op_ast_shape_matches_assumption(r)
    test_shipped_examples_declared_defaults(r)
    test_bare_cook_omitted_negative_param_uses_default(r)
    test_bare_cook_omitted_param_matches_explicit_default(r)
    test_tex_node_widget_defaults_unchanged(r)

    # ASK-6c: select(cond, a, b) — the NaN-isolation and CUDA-graph capturability rows
    # plus the reserved-name row; the other rows land as edits to the existing test
    # files wired above (test_codegen_optimizer's equivalence corpus, test_type_checker's
    # promoted typing + E5003, test_v018_precision's C1 decline, test_lazy_cooking's
    # never-sever row).
    from test_v036_ask6c import (
        test_ask6c_nan_untaken_arm_does_not_leak,
        test_ask6c_reserved_name_e3011,
        test_ask6c_captures_where_uniform_if_would_not,
    )
    test_ask6c_nan_untaken_arm_does_not_leak(r)
    test_ask6c_reserved_name_e3011(r)
    test_ask6c_captures_where_uniform_if_would_not(r)

    # TRK-25: a program whose control flow reduces a per-pixel value over the cooked region
    # is cooked WHOLE — strips, windows and batch strips are all declined — and the shapes
    # that split correctly (static/uniform loops, break/continue under a per-pixel guard)
    # keep splitting.
    from test_v036_region_dependence import (
        test_t1_repro_is_region_dependent,
        test_t7_predicate_is_not_a_blanket_disable,
        test_t7_no_shipped_program_is_declined,
        test_t11_analysis_failure_declines_the_split,
        test_t3_roi_window_is_declined,
        test_t5_halo_strip_route_is_declined,
        test_t4_batch_strips_are_declined,
        test_t6_string_merge_is_region_dependent,
        test_t6_string_ternary_is_region_dependent,
        test_t6_string_wires_need_the_binding_types,
        test_t6_binding_types_move_the_verdict_and_the_memo,
        test_t6_ternary_rule_is_not_a_blanket_disable,
        test_t2_strip_planner_declines,
        test_t9_gate_is_never_reached_on_an_unpressured_cook,
        test_t8_pragma_sunsets_the_loop_clause_only,
        test_t10_w7008_names_what_the_engine_now_refuses,
        test_t12_corpus_neutrality,
    )
    test_t1_repro_is_region_dependent(r)
    test_t7_predicate_is_not_a_blanket_disable(r)
    test_t7_no_shipped_program_is_declined(r)
    test_t11_analysis_failure_declines_the_split(r)
    test_t3_roi_window_is_declined(r)
    test_t5_halo_strip_route_is_declined(r)
    test_t4_batch_strips_are_declined(r)
    test_t6_string_merge_is_region_dependent(r)
    test_t6_string_ternary_is_region_dependent(r)
    test_t6_string_wires_need_the_binding_types(r)
    test_t6_binding_types_move_the_verdict_and_the_memo(r)
    test_t6_ternary_rule_is_not_a_blanket_disable(r)
    test_t2_strip_planner_declines(r)
    test_t9_gate_is_never_reached_on_an_unpressured_cook(r)
    test_t8_pragma_sunsets_the_loop_clause_only(r)
    test_t10_w7008_names_what_the_engine_now_refuses(r)
    test_t12_corpus_neutrality(r)

    # ENG-14: the two leaves split out of tex_engine.py (the frame-handoff/ownership
    # contract and the cook-fit planners), and the properties that keep them split —
    # the modules exist, every moved name defines in its new home and re-exports as a
    # real tex_engine global, both new modules stay leaves, and the tex_engine <->
    # tex_memory edge is still function-local.
    from test_eng14_split import (
        test_eng14_the_two_leaves_exist,
        test_eng14_the_moved_names_live_in_their_new_homes,
        test_eng14_the_new_modules_stay_leaves,
        test_eng14_tex_engine_still_imports_tex_memory_lazily,
    )
    test_eng14_the_two_leaves_exist(r)
    test_eng14_the_moved_names_live_in_their_new_homes(r)
    test_eng14_the_new_modules_stay_leaves(r)
    test_eng14_tex_engine_still_imports_tex_memory_lazily(r)

    # MUT-1: the mutation harness's curated suite list is DERIVED from its rows (each row
    # carries the suite that kills it) instead of hand-kept, so it cannot drift past a
    # release again — the drift that made all three TRK-25 rows assert nothing.
    from test_mut1_harness import (
        test_mut1_the_harness_imports_without_running_the_sweep,
        test_mut1_every_row_names_an_existing_killing_suite,
        test_mut1_a_row_without_a_usable_suite_is_refused_by_name,
        test_mut1_the_runner_import_list_is_derived_not_hand_written,
        test_mut1_every_rows_suite_is_loaded_by_the_runner,
    )
    test_mut1_the_harness_imports_without_running_the_sweep(r)
    test_mut1_every_row_names_an_existing_killing_suite(r)
    test_mut1_a_row_without_a_usable_suite_is_refused_by_name(r)
    test_mut1_the_runner_import_list_is_derived_not_hand_written(r)
    test_mut1_every_rows_suite_is_loaded_by_the_runner(r)

    # PUB-1: the archive is the product — `.comfyignore` keeps development material out of
    # the registry upload; every top-level directory is ignored or allowlisted by name; the
    # scanner-family sites over the SHIPPED files are pinned; no shipped module imports from
    # an ignored directory; validate-hw's triton lane SKIPs without benchmarks/.
    from test_pub1_archive import (
        test_pub1_every_top_level_directory_is_ignored_or_allowlisted,
        test_pub1_shipped_surface_ratchet,
        test_pub1_no_shipped_module_imports_an_ignored_directory,
        test_pub1_validate_hw_triton_lane_skips_without_benchmarks,
    )
    test_pub1_every_top_level_directory_is_ignored_or_allowlisted(r)
    test_pub1_shipped_surface_ratchet(r)
    test_pub1_no_shipped_module_imports_an_ignored_directory(r)
    test_pub1_validate_hw_triton_lane_skips_without_benchmarks(r)

    # DATA-6 L-D: EXR layer grouping — `read_layers` / `write_layers` over the DATA-2 reader,
    # the file half of PM-10. The grouping rule (last-dot split, root R/G/B/A -> `beauty`,
    # bare names -> own planes, R,G,B,A order, >4 / UINT refused) and the bitwise per-plane
    # round-trip; multipart / deep / tiled stay refused.
    from test_v037_exr_layers import (
        test_two_layers_and_a_bare_Z_round_trip_bitwise,
        test_grouping_splits_on_the_last_dot_and_orders_rgba,
        test_nuke_long_spelling_orders_rgb_and_round_trips_its_own_names,
        test_bare_names_are_own_planes_and_root_rgba_is_beauty,
        test_beauty_layer_vs_root_precedence,
        test_more_than_four_channels_is_refused_naming_the_layer,
        test_uint_channel_is_refused_loudly,
        test_multipart_deep_tiled_still_refused,
        test_write_layers_input_contract,
    )
    test_two_layers_and_a_bare_Z_round_trip_bitwise(r)
    test_grouping_splits_on_the_last_dot_and_orders_rgba(r)
    test_nuke_long_spelling_orders_rgb_and_round_trips_its_own_names(r)
    test_bare_names_are_own_planes_and_root_rgba_is_beauty(r)
    test_beauty_layer_vs_root_precedence(r)
    test_more_than_four_channels_is_refused_naming_the_layer(r)
    test_uint_channel_is_refused_loudly(r)
    test_multipart_deep_tiled_still_refused(r)
    test_write_layers_input_contract(r)

    # DATA-6 (compiler half): the lexer reads `@name.seg` as one token (one segment), the
    # splitback pass in `compile_ast` puts every non-plane dotted binding back to the swizzle
    # it always was, PLANES is a wire-only type inert in every expression rule and gated on the
    # engine profile, and the collision set is the 38 lowercase channel/swizzle names.
    from test_v037_planes_lexer import (
        test_dotted_at_binding_is_one_token,
        test_default_is_greedy_and_the_splitback_restores_the_ast,
        test_the_production_seam_lexes_greedily,
        test_one_segment_rule,
        test_p_prefix_declares_a_planes_wire,
        test_collision_set_is_38_lowercase_names,
        test_planes_is_inert_in_every_expression_rule,
        test_untyped_base_splits_back_to_a_swizzle,
        test_splitback_rows_and_their_mutations,
        test_splitback_is_an_identity_on_the_cook,
        test_swizzle_sugar_stays_refused,
        test_tripwire_is_portable,
    )
    test_dotted_at_binding_is_one_token(r)
    test_default_is_greedy_and_the_splitback_restores_the_ast(r)
    test_the_production_seam_lexes_greedily(r)
    test_one_segment_rule(r)
    test_p_prefix_declares_a_planes_wire(r)
    test_collision_set_is_38_lowercase_names(r)
    test_planes_is_inert_in_every_expression_rule(r)
    test_untyped_base_splits_back_to_a_swizzle(r)
    test_splitback_rows_and_their_mutations(r)
    test_splitback_is_an_identity_on_the_cook(r)
    test_swizzle_sugar_stays_refused(r)
    test_tripwire_is_portable(r)

    # DATA-6 (one front end): every consumer that reads a binding's name as the wire it is
    # connected to parses through `tex_cache.parse_and_split`; these pin that they all agree
    # on the wires a program reads, that a `p@` hint resolves to the same base everywhere, and
    # that the sigil scan is greedy while the identity map keeps the dotted base.
    from test_v037_frontend_parity import (
        test_every_front_end_agrees_on_the_wires_a_program_reads,
        test_hinted_plane_reads_resolve_the_same_base_everywhere,
        test_sigil_names_is_greedy_and_the_wire_keyed_consumers_keep_the_base,
    )
    test_every_front_end_agrees_on_the_wires_a_program_reads(r)
    test_hinted_plane_reads_resolve_the_same_base_everywhere(r)
    test_sigil_names_is_greedy_and_the_wire_keyed_consumers_keep_the_base(r)

    # DATA-6 (the wire): `PlanesValue`, demand-driven expansion at the engine seam, the
    # CACHE-6 key arms, the W7009 / E3304 / E7005 / E6003 refusals, the DATA-1 fan-out, the
    # fusion export refusal, the ComfyUI-profile invisibility and the deferred plane write.
    from test_v037_planes_wire import (
        test_a_planes_wire_expands_only_the_mentioned_planes,
        test_boundary_key_moves_when_an_unread_plane_changes,
        test_plane_cook_is_bit_exact_across_tiers,
        test_pm10_laziness_an_unread_plane_is_never_marshalled,
        test_undeclared_plane_is_w7009_with_a_did_you_mean,
        test_collision_is_e3304_and_Z_does_not_collide,
        test_unread_declared_plane_is_silent_and_w7002_stays_quiet,
        test_raw_dict_stays_e7005_with_a_planesvalue_hint,
        test_meta_fans_out_to_expanded_planes,
        test_e6003_names_the_slot_for_a_plane_read,
        test_fusion_refuses_a_dotted_export,
        test_planes_are_invisible_under_the_comfy_profile,
        test_plane_write_is_a_compile_error_naming_the_deferral,
        test_harness_prepares_a_plane_program,
        test_expansion_mutations,
    )
    test_a_planes_wire_expands_only_the_mentioned_planes(r)
    test_boundary_key_moves_when_an_unread_plane_changes(r)
    test_plane_cook_is_bit_exact_across_tiers(r)
    test_pm10_laziness_an_unread_plane_is_never_marshalled(r)
    test_undeclared_plane_is_w7009_with_a_did_you_mean(r)
    test_collision_is_e3304_and_Z_does_not_collide(r)
    test_unread_declared_plane_is_silent_and_w7002_stays_quiet(r)
    test_raw_dict_stays_e7005_with_a_planesvalue_hint(r)
    test_meta_fans_out_to_expanded_planes(r)
    test_e6003_names_the_slot_for_a_plane_read(r)
    test_fusion_refuses_a_dotted_export(r)
    test_planes_are_invisible_under_the_comfy_profile(r)
    test_plane_write_is_a_compile_error_naming_the_deferral(r)
    test_harness_prepares_a_plane_program(r)
    test_expansion_mutations(r)

    # DATA-6 (freeze #2): the language-version satellites a machine can read track
    # `tex_api.LANGUAGE_VERSION` — CF-7's JS pin generalised to the five stock manifests
    # and the generator that writes them.
    from test_v037_satellites import test_v037_language_version_satellites
    test_v037_language_version_satellites(r)

    # BENCH-2: the per-tick STRUCTURAL counts of the interactive host paths, pinned as exact
    # integers (timing cannot gate — see docs/host-path-counts.md and roadmap §10 item 3).
    # The CUDA row skips without a device; the mutation guard proves the counters can fire.
    from test_bench2_counts import (
        test_bench2_interactive_per_tick_counts,
        test_bench2_no_engine_side_cuda_sync_on_an_interactive_tick,
        test_bench2_cuda_per_tick_counts,
        test_bench2_free_memory_queries_per_tick,
        test_bench2_counters_are_not_inert,
    )
    test_bench2_interactive_per_tick_counts(r)
    test_bench2_no_engine_side_cuda_sync_on_an_interactive_tick(r)
    test_bench2_cuda_per_tick_counts(r)
    test_bench2_free_memory_queries_per_tick(r)
    test_bench2_counters_are_not_inert(r)

    # PERF-1: the ROI walk parses a source ONCE (the fold works on an `ast_nodes.clone_tree`
    # copy of a memoized parse) and answers exactly what the pre-change walk answered — the
    # oracle runs both over every shipped example and a corpus of value-sensitive shapes.
    from test_perf1_roi_walk_memo import (
        test_perf1_walk_answers_are_identical,
        test_perf1_a_source_is_parsed_once,
        test_perf1_the_clone_is_load_bearing,
        test_perf1_oracle_sensitive_rows,
        test_perf1_the_memo_hands_out_no_shared_ast,
        test_perf1_clone_tree_is_a_faithful_copy,
        test_perf1_nan_and_inf_do_not_become_a_radius,
    )
    test_perf1_walk_answers_are_identical(r)
    test_perf1_a_source_is_parsed_once(r)
    test_perf1_the_clone_is_load_bearing(r)
    test_perf1_oracle_sensitive_rows(r)
    test_perf1_the_memo_hands_out_no_shared_ast(r)
    test_perf1_clone_tree_is_a_faithful_copy(r)
    test_perf1_nan_and_inf_do_not_become_a_radius(r)

    # PERF-2: a builtin that needs a Python number (a kernel radius, an iteration count, a
    # flag) takes it from the host value the literal / `$param` was minted from, instead of
    # draining it off the device — bit-exact on both tiers, both devices, every sigma shape,
    # with the readback kept for a scalar that only exists on the device.
    from test_perf2_host_scalar import (
        test_perf2_outputs_are_bit_exact_on_both_tiers,
        test_perf2_convolve_and_patch_dist_are_bit_exact,
        test_perf2_the_two_tiers_still_agree,
        test_perf2_a_host_scalar_costs_no_readback,
        test_perf2_a_computed_scalar_still_reads_back,
        test_perf2_the_tag_carries_the_rounded_value,
        test_perf2_a_tag_never_survives_an_operation,
    )
    test_perf2_outputs_are_bit_exact_on_both_tiers(r)
    test_perf2_convolve_and_patch_dist_are_bit_exact(r)
    test_perf2_the_two_tiers_still_agree(r)
    test_perf2_a_host_scalar_costs_no_readback(r)
    test_perf2_a_computed_scalar_still_reads_back(r)
    test_perf2_the_tag_carries_the_rounded_value(r)
    test_perf2_a_tag_never_survives_an_operation(r)

    # PERF-3: the Gaussian kernel cache keys on the sigma the kernel is BUILT from, so the
    # same program with the same bindings answers the same whatever the process blurred
    # earlier — it used to key on `round(sigma, 3)` and build from the full value.
    from test_perf3_gauss_kernel_key import (
        test_perf3_a_blur_does_not_depend_on_what_was_blurred_first,
        test_perf3_the_cache_is_keyed_on_the_sigma_it_builds_from,
        test_perf3_a_repeated_sigma_still_hits_the_cache,
        test_perf3_the_rounded_key_is_detected,
        test_perf3_the_radius_rule_is_the_one_the_key_assumes,
    )
    test_perf3_a_blur_does_not_depend_on_what_was_blurred_first(r)
    test_perf3_the_cache_is_keyed_on_the_sigma_it_builds_from(r)
    test_perf3_a_repeated_sigma_still_hits_the_cache(r)
    test_perf3_the_rounded_key_is_detected(r)
    test_perf3_the_radius_rule_is_the_one_the_key_assumes(r)

    # PERF-4: the two remaining front-end re-scans. The lazy analysis parses a source ONCE
    # (its fold works on an `ast_nodes.clone_tree` copy of a memoized parse) and the ROI
    # halo scan traverses the program once instead of twice-plus-a-re-descent — both proved
    # against the pre-change implementations over every shipped example and the comp.
    from test_perf4_front_end_rescans import (
        test_perf4_lazy_answers_are_identical,
        test_perf4_a_lazy_source_is_parsed_once,
        test_perf4_the_lazy_clone_is_load_bearing,
        test_perf4_the_lazy_memo_hands_out_no_shared_ast,
        test_perf4_lazy_oracle_sensitive_rows,
        test_perf4_halo_answers_are_identical,
        test_perf4_halo_corpus_is_not_vacuous,
        test_perf4_the_halo_scan_visits_each_node_once,
        test_perf4_halo_mutants_are_caught,
    )
    test_perf4_lazy_answers_are_identical(r)
    test_perf4_a_lazy_source_is_parsed_once(r)
    test_perf4_the_lazy_clone_is_load_bearing(r)
    test_perf4_the_lazy_memo_hands_out_no_shared_ast(r)
    test_perf4_lazy_oracle_sensitive_rows(r)
    test_perf4_halo_answers_are_identical(r)
    test_perf4_halo_corpus_is_not_vacuous(r)
    test_perf4_the_halo_scan_visits_each_node_once(r)
    test_perf4_halo_mutants_are_caught(r)

    # PERF-5: the fixed per-cook front end. One `TEXCache.fingerprint` per cook (the key
    # `prepare` needs anyway is handed to the compile, which uses it for the probe AND the
    # store), guarded by a base-sha golden over every shipped example's fingerprint string —
    # the string names an on-disk cache file, so it may not move.
    from test_perf5_fixed_pipeline import (
        test_perf5_fingerprint_and_param_only_names_are_byte_stable,
        test_perf5_the_golden_catches_a_one_character_recipe_change,
        test_perf5_one_fingerprint_per_cook,
        test_perf5_the_shared_fingerprint_is_the_cache_key,
        test_perf5_one_lex_per_never_seen_program,
        test_perf5_a_claimed_stream_parses_to_the_same_program,
        test_perf5_the_token_handoff_is_consumed,
        test_perf5_a_lex_failure_offers_nothing,
    )
    test_perf5_fingerprint_and_param_only_names_are_byte_stable(r)
    test_perf5_the_golden_catches_a_one_character_recipe_change(r)
    test_perf5_one_fingerprint_per_cook(r)
    test_perf5_the_shared_fingerprint_is_the_cache_key(r)
    test_perf5_one_lex_per_never_seen_program(r)
    test_perf5_a_claimed_stream_parses_to_the_same_program(r)
    test_perf5_the_token_handoff_is_consumed(r)
    test_perf5_a_lex_failure_offers_nothing(r)

    # PERF-6: the free-VRAM reading the cook-fit planners divide. One live host query per
    # unpressured frame instead of one per cook, guarded by a base-sha golden over the whole
    # example corpus's tile plans — the mechanism is only allowed to change what the engine
    # PAYS, never what it plans.
    from test_perf6_free_memory_once import (
        test_perf6_the_tile_plans_are_identical,
        test_perf6_the_golden_catches_a_planner_change,
        test_perf6_an_unpressured_frame_asks_the_host_once,
        test_perf6_a_pressured_cook_always_reads_live,
        test_perf6_the_memo_margin_is_load_bearing,
        test_perf6_the_bound_tracks_the_allocator,
        test_perf6_a_swapped_host_is_asked_afresh,
    )
    test_perf6_the_tile_plans_are_identical(r)
    test_perf6_the_golden_catches_a_planner_change(r)
    test_perf6_an_unpressured_frame_asks_the_host_once(r)
    test_perf6_a_pressured_cook_always_reads_live(r)
    test_perf6_the_memo_margin_is_load_bearing(r)
    test_perf6_the_bound_tracks_the_allocator(r)
    test_perf6_a_swapped_host_is_asked_afresh(r)

    # PERF-7: what a COLD cook on the compiled tier is allowed to do, as counts. The
    # bench's `cpu_on_cold` config times exactly `clear_compiled_cache()` +
    # `execute_compiled`, and its geomean is inside this box's noise floor
    # (docs/roadmap.md §10 item 3), so the contract is pinned as frames instead: no
    # re-lex, no re-parse, no re-fold, no re-emit, no re-tag, and the compile route
    # entered exactly once.
    from test_perf7_compiled_cold import (
        test_perf7_cold_compiled_cook_does_no_front_end_work,
        test_perf7_cold_compiled_cook_runs_the_route_once,
        test_perf7_cold_compiled_cook_frame_ceiling,
        test_perf7_the_counter_is_not_inert,
    )
    test_perf7_cold_compiled_cook_does_no_front_end_work(r)
    test_perf7_cold_compiled_cook_runs_the_route_once(r)
    test_perf7_cold_compiled_cook_frame_ceiling(r)
    test_perf7_the_counter_is_not_inert(r)

    # PERF-8: the five analysis memos carry the egress profile, because the front end they
    # derive from parses `p@beauty.diffuse` as a plane read or a swizzle depending on it.
    from test_perf8_memo_flag_key import (
        test_perf8_a_parse_memo_entry_is_not_served_across_a_profile_flip,
        test_perf8_the_analysis_answers_follow_the_profile,
        test_perf8_dropping_the_flag_from_the_key_brings_the_hazard_back,
        test_perf8_one_shared_body_bounded_and_clearable,
    )
    test_perf8_a_parse_memo_entry_is_not_served_across_a_profile_flip(r)
    test_perf8_the_analysis_answers_follow_the_profile(r)
    test_perf8_dropping_the_flag_from_the_key_brings_the_hazard_back(r)
    test_perf8_one_shared_body_bounded_and_clearable(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
