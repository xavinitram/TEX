#!/usr/bin/env python
"""Standalone TEX test runner — python tests/run_all.py"""
from helpers import SubTestResult

from test_lexer import test_lexer, test_lexer_v11
from test_parser import test_parser, test_parser_v11
from test_type_checker import test_type_checker
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
    test_node_helpers,
)
from test_codegen_optimizer import (
    test_codegen_equivalence, test_optimization_regressions, test_licm,
    test_optimizer_passes,
)


def main():
    print("=" * 60)
    print("TEX Test Suite")
    print("=" * 60)

    r = SubTestResult()

    test_lexer(r)
    test_lexer_v11(r)
    test_parser(r)
    test_parser_v11(r)
    test_type_checker(r)
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
    test_node_helpers(r)

    success = r.summary()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
