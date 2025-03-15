import logging
import os
import traceback
from typing import Any, Dict, List

# Import services for various analysis aspects:
# - Data flow analysis and function similarity comparisons.
from services.data_flow_analyzer import analyze_data_flow, compare_function_similarity
# - Refactoring planner: to generate refactoring suggestions and structured plans.
from services.refactoring_planner import RefactoringSug, RefactoringPlanner, RefactoringPlan
# - Token-based clone detection service.
from services.token_based_det import TokenBasedCloneDetector
# - Analysis services for dynamic importing, static analysis, and runtime analysis.
from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    RuntimeAnalysisService,
)
# NEW: Import the Rope refactoring engine for applying structured refactorings.
from services.rope_refactor_engine import RopeRefactorEngine

# Configure logging to display time, level, and message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class CodeAnalysisController:
    """
    Orchestrates the entire analysis pipeline:
      - Dynamically imports Python files.
      - Performs static analysis (AST-based clone detection and complexity analysis).
      - Executes token-based clone detection.
      - Analyzes data flow between functions.
      - Runs runtime checks to gather performance metrics.
      - Merges analysis results from different sources.
      - Generates both textual refactoring suggestions and structured refactoring plans.

    In addition, it manages refactoring through:
      - A RefactoringPlanner to produce plans.
      - A RopeRefactorEngine to apply these plans.
    """

    def __init__(self):
        # Global namespace holds dynamically imported function definitions.
        self.global_namespace: Dict[str, Any] = {}
        # Instantiate the runtime analysis service.
        self.runtime_analysis = RuntimeAnalysisService()

        # Instantiate the refactoring planner for generating structured plans.
        self.refactoring_planner = RefactoringPlanner()

        # The Rope refactoring engine will be initialized later when the project path is known.
        self.rope_engine = None

    def set_project_path(self, project_path: str) -> None:
        """
        Initializes or re-initializes the RopeRefactorEngine with the user-provided project path.
        This method should be called by the GUI once it determines the folder or file location.
        """
        if not os.path.isdir(project_path):
            logging.warning(f"Project path {project_path} is not a valid directory.")
        self.rope_engine = RopeRefactorEngine(project_path)
        logging.info(f"RopeRefactorEngine initialized for path: {project_path}")

    def get_python_files(self, path: str) -> List[str]:
        """
        Retrieves a list of Python (.py) files from the given path.

        If 'path' is a directory, recursively walks the directory and collects all .py files.
        Otherwise, returns a list containing the single file path.
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
        Main entry point for analyzing code. This method coordinates multiple analysis steps
        and returns a comprehensive results dictionary containing:
          - 'static': AST-based clone pairs and static complexity information.
          - 'token_clones': Token-based clone detection results.
          - 'merged_clones': Merged results from AST, token, data flow, and runtime analyses.
          - 'runtime': Runtime metrics for function execution.
          - 'data_flow': Data flow analysis results.
          - 'refactoring_suggestions': Textual refactoring suggestions.
          - 'refactoring_plans': Structured refactoring plans.
          - 'analysis_insights': Additional insights (e.g., top duplicates, slowest functions).
        """
        try:
            # 1. Collect Python files from the specified path.
            python_files = self.get_python_files(path)
            # Clear the global namespace before dynamic import.
            self.global_namespace.clear()

            # Dynamically import each file and add callable objects to the global namespace.
            for file in python_files:
                module = DynamicImportService.dynamic_import(file)
                if module:
                    for name in dir(module):
                        obj = getattr(module, name)
                        if callable(obj):
                            self.global_namespace[name] = obj

            # 2. Perform Static Analysis:
            # Parse files into ASTs and identify similar nodes using the static analysis service.
            ast_trees, static_nodes, _, function_line_map = StaticAnalysisService.parse_and_find_similarities(
                python_files, threshold
            )
            # Filter out invalid static node pairs with missing or empty function names.
            static_nodes = [
                item for item in static_nodes
                if item.get("function1_metrics", {}).get("name", "").strip()
                   and item.get("function2_metrics", {}).get("name", "").strip()
            ]

            # 3. Execute Token-Based Clone Detection:
            token_clones = []
            for file in python_files:
                token_clones.extend(
                    TokenBasedCloneDetector.detect_token_clones_with_ast(file, 0.0)
                )
            # Filter out token clones with missing function names.
            token_clones = [
                clone for clone in token_clones
                if clone.get("func1", "").strip() and clone.get("func2", "").strip()
            ]

            # 4. Conduct Data Flow Analysis:
            data_flow_results = analyze_data_flow(ast_trees)

            # 5. Execute Runtime Analysis:
            # Inject function line ranges for use by the runtime analyzer.
            self.runtime_analysis.runtime_analyzer.function_line_map = function_line_map
            # Apply runtime tests to both static node pairs and token clones.
            self.runtime_analysis.apply_runtime_checks(static_nodes, self.global_namespace, num_tests)
            self.runtime_analysis.apply_runtime_checks(token_clones, self.global_namespace, num_tests)
            runtime_results = self.runtime_analysis.get_runtime_data()

            # 6. Merge Data from All Analyses:
            merged_clones = self._merge_ast_token_dataflow(
                static_nodes,
                token_clones,
                data_flow_results,
                runtime_results
            )

            # 7. Generate Textual Refactoring Suggestions:
            refactoring_suggestions = RefactoringSug.generate_suggestions(merged_clones, data_flow_results)

            # 7.5 Convert suggestions into structured RefactoringPlans.
            refactoring_plans = self.refactoring_planner.create_plans_from_suggestions(
                merged_clones,
                refactoring_suggestions
            )

            # 8. Compute Additional Insights (e.g., top complex functions, slowest functions, duplicate pairs):
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
            }

            # 9. Return the aggregated results.
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
        Applies user-selected refactoring plans.

        This method is invoked by the GUI (e.g., via the RefactorTab) and uses
        the RopeRefactorEngine to execute the refactoring plans.
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
        Merges data from multiple analyses (AST, token-based, data flow, runtime, static complexity)
        into a single list of merged clone pairs.

        """
        merged_map = {}

        # 1. Build merged entries from AST-based pairs.
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
            }

        # 2. Merge token-based clone data into the merged map.
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
                }

        # 3. Merge Data Flow Similarity and inject additional function-level details.
        for pair_key, pair_data in merged_map.items():
            df_sim = self._compute_data_flow_similarity(
                pair_data["func1"],
                pair_data["func2"],
                data_flow
            )
            pair_data["dataflow_similarity"] = df_sim

            self._inject_extra_func_info(pair_data["function1_metrics"], data_flow)
            self._inject_extra_func_info(pair_data["function2_metrics"], data_flow)

        # 4. Compute Runtime Similarity based on runtime metrics.
        for pair_key, pair_data in merged_map.items():
            rt_sim = self._compute_runtime_similarity(
                pair_data["func1"],
                pair_data["func2"],
                runtime_results
            )
            pair_data["runtime_similarity"] = rt_sim

        return list(merged_map.values())

    def _compute_data_flow_similarity(self, func1: str, func2: str, data_flow: Dict[str, Any]) -> float:
        """
        Computes similarity between two functions based on their data flow dependencies.

        It retrieves the data flow info for each function and compares them using a similarity function.
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
        Computes similarity between two functions based on runtime metrics such as average execution time.
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

    def _inject_extra_func_info(self, func_metrics: Dict[str, Any], data_flow: Dict[str, Any]) -> None:
        """
        Enhances function metrics by injecting extra information from data flow analysis,
        such as dependency calls and inferred return types.
        """
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
