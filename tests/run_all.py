#!/usr/bin/env python
"""Standalone TEX test runner — python tests/run_all.py"""
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
from test_release_gate import test_version_consistency, test_codegen_determinism
from test_v017_phase1 import (
    test_tst5_tier_trace,
    test_tst6_registry_parity,
    test_tst2_edge_matrix,
    test_tst4_operator_completeness,
    test_tst1_differential_fuzzer,
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
from test_v018_docs import test_doc7b_map_drift, test_reg1b_doc_ex_populated


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
    test_doc7b_map_drift(r)
    test_reg1b_doc_ex_populated(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
