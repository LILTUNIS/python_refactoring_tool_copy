import logging
import os
import traceback
from typing import Any, Dict, List

# Existing imports
from services.data_flow_analyzer import analyze_data_flow, compare_function_similarity
from services.refactoring_planner import RefactoringSug, RefactoringPlanner, RefactoringPlan
from services.token_based_det import TokenBasedCloneDetector
from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    RuntimeAnalysisService,
    DashboardMetricsService
)
# NEW: Import the RopeRefactorEngine here
from services.rope_refactor_engine import RopeRefactorEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CodeAnalysisController:
    """
    Orchestrates dynamic import, static analysis, token clone detection,
    data flow analysis, runtime checks, and dashboard metrics computation.

    Now also manages refactoring:
      - A RefactoringPlanner to create structured plans
      - A RopeRefactorEngine to apply those plans
    """

    def __init__(self):
        self.global_namespace: Dict[str, Any] = {}
        self.runtime_analysis = RuntimeAnalysisService()

        # Instantiate the planner (for generating structured plans)
        self.refactoring_planner = RefactoringPlanner()

        # We will initialize the rope engine later, once we know the project path.
        # If your tool uses a default path, you can set it here.
        self.rope_engine = None

    def set_project_path(self, project_path: str) -> None:
        """
        Initialize or re-initialize the RopeRefactorEngine with the user’s project path.
        The GUI can call this once it knows the folder or file location.
        """
        if not os.path.isdir(project_path):
            logging.warning(f"Project path {project_path} is not a valid directory.")
        self.rope_engine = RopeRefactorEngine(project_path)
        logging.info(f"RopeRefactorEngine initialized for path: {project_path}")

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
          - 'static'  => AST-based pairs (with AST similarity & static complexity data)
          - 'token_clones'
          - 'merged_clones'
          - 'runtime'
          - 'data_flow'
          - 'refactoring_suggestions' => textual suggestions
          - 'refactoring_plans'      => structured plans
          - 'analysis_insights'      => top duplicates, slowest, etc.
        """
        try:
            # 1. Gather Python files and dynamically import them
            python_files = self.get_python_files(path)
            self.global_namespace.clear()

            for file in python_files:
                module = DynamicImportService.dynamic_import(file)
                if module:
                    for name in dir(module):
                        obj = getattr(module, name)
                        if callable(obj):
                            self.global_namespace[name] = obj

            # 2. Static Analysis
            ast_trees, static_nodes, _, function_line_map = StaticAnalysisService.parse_and_find_similarities(
                python_files, threshold
            )
            # Filter out invalid pairs
            static_nodes = [
                item for item in static_nodes
                if item.get("function1_metrics", {}).get("name", "").strip()
                   and item.get("function2_metrics", {}).get("name", "").strip()
            ]

            # 3. Token-Based Clone Detection
            token_clones = []
            for file in python_files:
                token_clones.extend(
                    TokenBasedCloneDetector.detect_token_clones_with_ast(file, 0.0)
                )
            token_clones = [
                clone for clone in token_clones
                if clone.get("func1", "").strip() and clone.get("func2", "").strip()
            ]

            # 4. Data Flow Analysis
            data_flow_results = analyze_data_flow(ast_trees)

            # 5. Runtime Analysis
            self.runtime_analysis.runtime_analyzer.function_line_map = function_line_map
            self.runtime_analysis.apply_runtime_checks(static_nodes, self.global_namespace, num_tests)
            self.runtime_analysis.apply_runtime_checks(token_clones, self.global_namespace, num_tests)
            runtime_results = self.runtime_analysis.get_runtime_data()

            # 6. Merge All Data
            merged_clones = self._merge_ast_token_dataflow(
                static_nodes,
                token_clones,
                data_flow_results,
                runtime_results
            )

            # 7. Generate Textual Refactoring Suggestions
            refactoring_suggestions = RefactoringSug.generate_suggestions(merged_clones, data_flow_results)

            # 7.5 Convert suggestions => structured RefactoringPlans
            refactoring_plans = self.refactoring_planner.create_plans_from_suggestions(
                merged_clones,
                refactoring_suggestions
            )

            # 8. Compute Dashboard Metrics
            dashboard_metrics = DashboardMetricsService.compute_key_metrics(
                static_nodes,
                token_clones,
                runtime_results
            )

            # 9. Additional Insights
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

            # 10. Return All Results
            return {
                "static": static_nodes,
                "token_clones": token_clones,
                "merged_clones": merged_clones,
                "runtime": runtime_results,
                "data_flow": data_flow_results,
                "refactoring_suggestions": refactoring_suggestions,
                "refactoring_plans": refactoring_plans,
                "analysis_insights": analysis_insights,
            }
        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Analysis failed: {e}\nTraceback:\n{tb_str}")
            raise RuntimeError(f"Error during analysis: {e}\n\nTraceback:\n{tb_str}")

    # --------------------------------------------------------------------------
    #                           Refactoring Application
    # --------------------------------------------------------------------------
    def apply_refactoring(self, plans: List[RefactoringPlan]) -> None:
        """
        Called by the GUI (RefactorTab) to apply user-selected refactoring plans.
        We rely on our RopeRefactorEngine (self.rope_engine).
        """
        if not self.rope_engine:
            logging.warning("No RopeRefactorEngine is initialized. Please call set_project_path().")
            return

        for plan in plans:
            try:
                self.rope_engine.apply_refactor_plan(plan)
                logging.info(f"Applied refactoring plan: {plan}")
            except Exception as ex:
                logging.error(f"Failed to apply plan {plan}: {ex}", exc_info=True)

    # --------------------------------------------------------------------------
    #                           HELPER METHODS
    # --------------------------------------------------------------------------
    def _merge_ast_token_dataflow(
        self,
        static_nodes: List[Dict[str, Any]],
        token_clones: List[Dict[str, Any]],
        data_flow: Dict[str, Any],
        runtime_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Merges AST-based data, token-based data, data flow, runtime,
        and static complexity info into a single 'merged_clones' list.
        """
        merged_map = {}

        # 1. Build from AST-based pairs
        for item in static_nodes:
            f1_info = item["function1_metrics"]
            f2_info = item["function2_metrics"]
            f1 = f1_info["name"].strip()
            f2 = f2_info["name"].strip()

            pair_key = tuple(sorted([f1, f2]))

            merged_map[pair_key] = {
                "func1": f1,
                "func2": f2,
                "function1_metrics": dict(f1_info),
                "function2_metrics": dict(f2_info),
                "ast_similarity": item.get("ast_similarity", 0.0),
                "type1_similarity": item.get("type1_similarity", 0.0),
                "token_similarity": None,
                "dataflow_similarity": 0.0,
                "runtime_similarity": 0.0,
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
            df_sim = self._compute_data_flow_similarity(
                pair_data["func1"],
                pair_data["func2"],
                data_flow
            )
            pair_data["dataflow_similarity"] = df_sim

            self._inject_extra_func_info(pair_data["function1_metrics"], data_flow)
            self._inject_extra_func_info(pair_data["function2_metrics"], data_flow)

        # 4. Runtime Similarity
        for pair_key, pair_data in merged_map.items():
            rt_sim = self._compute_runtime_similarity(
                pair_data["func1"],
                pair_data["func2"],
                runtime_results
            )
            pair_data["runtime_similarity"] = rt_sim

        return list(merged_map.values())

    def _compute_data_flow_similarity(self, func1: str, func2: str, data_flow: Dict[str, Any]) -> float:
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
        return round(1.0 - abs(s1 - s2), 3)

    def _inject_extra_func_info(self, func_metrics: Dict[str, Any], data_flow: Dict[str, Any]) -> None:
        fname = func_metrics.get("name", "")
        if not fname:
            return

        for file_path, funcs in data_flow.items():
            if fname in funcs:
                deps = funcs[fname].get("dependencies", {})
                calls = deps.get("function_calls", [])
                func_metrics["calls"] = calls

                returns = deps.get("returns", [])
                if returns:
                    if any("str" in r.lower() for r in returns):
                        func_metrics["return_type"] = "str"
                    elif any("[" in r or "]" in r for r in returns):
                        func_metrics["return_type"] = "list"
                    else:
                        func_metrics["return_type"] = "int"
                else:
                    func_metrics["return_type"] = "Unknown"
                break
