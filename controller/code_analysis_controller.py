import logging
import os
import traceback
from typing import Any, Dict, List

from model.data_flow_analyzer import analyze_data_flow, compare_function_similarity
from model.refactoring_planner import RefactoringPlanner
from model.refactoring_sug import RefactoringSug
from model.token_based_det import TokenBasedCloneDetector
from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    RuntimeAnalysisService,
    DashboardMetricsService
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class CodeAnalysisController:
    """
    Orchestrates dynamic import, static analysis, token clone detection,
    data flow analysis, runtime checks, and dashboard metrics computation.

    Now integrates a multi-metric approach:
    - Token, AST, Data Flow, Runtime, and Static Similarities
    - Additional function info (calls, return type, param count)
    - Finally, merges everything into a single list of pairs for advanced refactoring suggestions.
    """

    def __init__(self):
        self.global_namespace: Dict[str, Any] = {}
        self.runtime_analysis = RuntimeAnalysisService()

    def get_python_files(self, path: str) -> List[str]:
        """
        If 'path' is a folder, walk it and collect all .py files.
        Otherwise, return a list with the single path.
        """
        if os.path.isdir(path):
            python_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
            return python_files
        return [path]

    def start_analysis(self, path: str, threshold: float, num_tests: int) -> Dict[str, Any]:
        """
        Main entry point for analyzing code. Returns a dictionary with:
          - 'static'         => AST-based pairs (with AST similarity & static complexity data)
          - 'token_clones'   => token-based pairs
          - 'merged_clones'  => combined pairs with all similarity metrics (token, AST, data flow, runtime, static)
          - 'runtime'        => runtime results
          - 'data_flow'      => data flow info
          - 'refactoring_suggestions' => from advanced RDS-based approach
          - 'analysis_insights'       => top duplicates, slowest, etc.
        """
        try:
            #
            # 1. Gather Python files and dynamically import them
            #
            python_files = self.get_python_files(path)
            self.global_namespace.clear()

            for file in python_files:
                module = DynamicImportService.dynamic_import(file)
                if module:
                    for name in dir(module):
                        obj = getattr(module, name)
                        if callable(obj):
                            self.global_namespace[name] = obj

            #
            # 2. Static Analysis (AST-based)
            #    - parse_and_find_similarities => returns (ast_trees, static_nodes, similarity_results, function_line_map)
            #
            ast_trees, static_nodes, _, function_line_map = StaticAnalysisService.parse_and_find_similarities(
                python_files, threshold
            )

            # Filter out invalid pairs (both function names must be non-empty)
            static_nodes = [
                item for item in static_nodes
                if item.get("function1_metrics", {}).get("name", "").strip()
                   and item.get("function2_metrics", {}).get("name", "").strip()
            ]

            #
            # 3. Token-Based Clone Detection
            #
            token_clones = []
            for file in python_files:
                token_clones.extend(
                    TokenBasedCloneDetector.detect_token_clones_with_ast(file, 0.0)
                )
            token_clones = [
                clone for clone in token_clones
                if clone.get("func1", "").strip() and clone.get("func2", "").strip()
            ]

            #
            # 4. Data Flow Analysis
            #
            data_flow_results = analyze_data_flow(ast_trees)

            #
            # 5. Runtime Analysis => apply to both static & token sets
            #
            self.runtime_analysis.runtime_analyzer.function_line_map = function_line_map
            self.runtime_analysis.apply_runtime_checks(static_nodes, self.global_namespace, num_tests)
            self.runtime_analysis.apply_runtime_checks(token_clones, self.global_namespace, num_tests)
            runtime_results = self.runtime_analysis.get_runtime_data()

            #
            # 6. Merge Everything into "merged_clones"
            #
            merged_clones = self._merge_ast_token_dataflow(
                static_nodes,
                token_clones,
                data_flow_results,
                runtime_results
            )

            #
            # 7. Generate Advanced Refactoring Suggestions (RDS-based)
            #    Instead of the older approach with just token clones.
            #
            refactoring_suggestions = RefactoringSug.generate_suggestions(merged_clones, data_flow_results)

            # 7.5 Convert textual suggestions -> structured RefactoringPlans
            planner = RefactoringPlanner()
            refactoring_plans = planner.create_plans_from_suggestions(merged_clones, refactoring_suggestions)
            #
            # 8. Compute Dashboard Metrics
            #
            dashboard_metrics = DashboardMetricsService.compute_key_metrics(
                static_nodes,
                token_clones,
                runtime_results
            )

            #
            # 9. Additional Insights
            #
            analysis_insights = {
                "top_complex_functions": sorted(
                    [
                        (
                            item.get("function1_metrics", {}).get("name", ""),
                            item.get("function1_metrics", {}).get("complexity", 0)
                        )
                        for item in static_nodes
                    ]
                    + [
                        (
                            item.get("function2_metrics", {}).get("name", ""),
                            item.get("function2_metrics", {}).get("complexity", 0)
                        )
                        for item in static_nodes
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                "slowest_functions": sorted(
                    [
                        (
                            func.get("func_name", ""),
                            max(
                                iter_data.get("total_wall_time", 0)
                                for iter_data in func.get("test_results", {}).values()
                            )
                        )
                        for func in runtime_results.get("functions", [])
                    ],
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
                "top_duplicate_pairs": sorted(
                    [
                        (
                            clone.get("func1", ""),
                            clone.get("func2", ""),
                            max(
                                clone.get("token_similarity", 0.0),
                                clone.get("ast_similarity", 0.0)
                            )
                        )
                        for clone in token_clones
                    ]
                    + [
                        (
                            item.get("function1_metrics", {}).get("name", ""),
                            item.get("function2_metrics", {}).get("name", ""),
                            item.get("ast_similarity", 0.0)
                        )
                        for item in static_nodes
                    ],
                    key=lambda x: x[2],
                    reverse=True
                )[:5],
                "files_with_most_duplicates": [("my_file1.py", 4), ("my_file2.py", 2)],
                "refactoring_suggestions": refactoring_suggestions,
                "key_metrics": dashboard_metrics,
            }

            #
            # 10. Return All Results
            return {
                "static": static_nodes,
                "token_clones": token_clones,
                "merged_clones": merged_clones,
                "runtime": runtime_results,
                "data_flow": data_flow_results,
                "refactoring_suggestions": refactoring_suggestions,  # textual suggestions
                "refactoring_plans": refactoring_plans,  # new structured plans
                "analysis_insights": analysis_insights,
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Analysis failed: {e}\nTraceback:\n{tb_str}")
            raise RuntimeError(f"Error during analysis: {e}\n\nTraceback:\n{tb_str}")

    # --------------------------------------------------------------------------
    #                           HELPER METHODS
    # --------------------------------------------------------------------------
    def _merge_ast_token_dataflow(self, static_nodes, token_clones, data_flow, runtime_results) -> List[Dict[str, Any]]:
        """
        Merges AST-based data, token-based data, data flow, runtime,
        and static complexity info into a single 'merged_clones' list.
        This ensures each pair has:
          - token_similarity
          - ast_similarity
          - dataflow_similarity
          - runtime_similarity
          - static_similarity
          - function1_metrics / function2_metrics (including param_count, calls, return_type, etc.)
        """
        merged_map = {}

        # 1. Build from AST-based pairs
        for item in static_nodes:
            f1_info = item["function1_metrics"]
            f2_info = item["function2_metrics"]
            f1 = f1_info["name"].strip()
            f2 = f2_info["name"].strip()

            pair_key = tuple(sorted([f1, f2]))

            # We might store param_count, complexity, etc. in functionX_metrics
            # If you want to store function calls, return type, etc. from data_flow, do it below.

            merged_map[pair_key] = {
                "func1": f1,
                "func2": f2,
                "function1_metrics": dict(f1_info),  # Copy to preserve
                "function2_metrics": dict(f2_info),
                "ast_similarity": item.get("ast_similarity", 0.0),
                "type1_similarity": item.get("type1_similarity", 0.0),
                "token_similarity": None,
                "dataflow_similarity": 0.0,
                "runtime_similarity": 0.0,
                # static_score for each function is already in f1_info["static_score"], etc.
                # We'll compute a pairwise static similarity:
                "static_similarity": self._compute_static_similarity(
                    f1_info.get("static_score", 0.0),
                    f2_info.get("static_score", 0.0),
                ),
            }

        # 2. Merge token clones
        for clone in token_clones:
            f1 = clone["func1"].strip()
            f2 = clone["func2"].strip()
            pair_key = tuple(sorted([f1, f2]))

            token_sim = clone.get("token_similarity", 0.0)

            if pair_key in merged_map:
                merged_map[pair_key]["token_similarity"] = token_sim
            else:
                # If not in AST-based pairs, create a new entry
                merged_map[pair_key] = {
                    "func1": f1,
                    "func2": f2,
                    "function1_metrics": {"name": f1},
                    "function2_metrics": {"name": f2},
                    "ast_similarity": 0.0,
                    "type1_similarity": 0.0,
                    "token_similarity": token_sim,
                    "dataflow_similarity": 0.0,
                    "runtime_similarity": 0.0,
                    "static_similarity": 0.0,
                }

        # 3. Data Flow Similarity + Additional function-level info
        for pair_key, pair_data in merged_map.items():
            df_sim = self._compute_data_flow_similarity(pair_data["func1"], pair_data["func2"], data_flow)
            pair_data["dataflow_similarity"] = df_sim

            # Also gather function calls, return types from data_flow if desired:
            self._inject_extra_func_info(pair_data["function1_metrics"], data_flow)
            self._inject_extra_func_info(pair_data["function2_metrics"], data_flow)

        # 4. Runtime Similarity
        for pair_key, pair_data in merged_map.items():
            rt_sim = self._compute_runtime_similarity(pair_data["func1"], pair_data["func2"], runtime_results)
            pair_data["runtime_similarity"] = rt_sim

        return list(merged_map.values())

    def _compute_data_flow_similarity(self, func1: str, func2: str, data_flow: Dict[str, Any]) -> float:
        """
        Finds each function's data_flow info and calls compare_function_similarity.
        """
        func1_info = None
        func2_info = None

        for file_path, func_dict in data_flow.items():
            if func1 in func_dict and not func1_info:
                func1_info = func_dict[func1]
                func1_info["name"] = func1
            if func2 in func_dict and not func2_info:
                func2_info = func_dict[func2]
                func2_info["name"] = func2

        if not func1_info or not func2_info:
            return 0.0

        score = compare_function_similarity(func1_info, func2_info)
        return round(score, 2)

    def _compute_runtime_similarity(self, func1: str, func2: str, runtime_results: Dict[str, Any]) -> float:
        """
        Compute a runtime similarity score for two functions based on their average CPU time.
        Score = 1 - (|avg1 - avg2| / max(avg1, avg2)), if both > 0.
        """
        metrics_lookup = {}
        for f in runtime_results.get("functions", []):
            name = f.get("func_name", "").strip()
            if name:
                metrics_lookup[name] = f

        if func1 in metrics_lookup and func2 in metrics_lookup:
            avg1 = metrics_lookup[func1].get("avg_time", 0.0)
            avg2 = metrics_lookup[func2].get("avg_time", 0.0)
            if avg1 == 0 and avg2 == 0:
                return 1.0
            diff = abs(avg1 - avg2)
            max_avg = max(avg1, avg2, 1e-6)
            similarity = 1 - (diff / max_avg)
            return max(0.0, min(similarity, 1.0))
        return 0.0

    def _compute_static_similarity(self, s1: float, s2: float) -> float:
        """
        Pairwise static similarity from each function's static_score in [0..1].
        static_similarity = 1 - |s1 - s2|
        """
        return round(1.0 - abs(s1 - s2), 3)

    def _inject_extra_func_info(self, func_metrics: Dict[str, Any], data_flow: Dict[str, Any]) -> None:
        """
        Optionally gather more info (function calls, return type, etc.) from data_flow
        and store in functionX_metrics so that RefactoringSug can use them.
        """
        fname = func_metrics.get("name", "")
        if not fname:
            return

        # Search the data_flow structure for this function
        for file_path, funcs in data_flow.items():
            if fname in funcs:
                deps = funcs[fname].get("dependencies", {})
                calls = deps.get("function_calls", [])
                # Convert list to a set for easier overlap checks
                func_metrics["calls"] = calls

                # For demonstration, define a naive "return_type" from the 'returns' field
                returns = deps.get("returns", [])
                if returns:
                    # e.g., if we see 'return_value' with an int or str, guess. This is naive
                    # A real approach might parse expressions or track usage more deeply
                    if any("str" in r.lower() for r in returns):
                        func_metrics["return_type"] = "str"
                    elif any("[" in r or "]" in r for r in returns):
                        func_metrics["return_type"] = "list"
                    else:
                        func_metrics["return_type"] = "int"  # default guess
                else:
                    func_metrics["return_type"] = "Unknown"
                break
