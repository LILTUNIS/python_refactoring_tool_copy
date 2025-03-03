import inspect
import logging
import os
import traceback
from typing import Any, Dict, List

from model.data_flow_analyzer import compare_function_similarity
from model.refactoring_sug import RefactoringSug
from model.token_based_det import TokenBasedCloneDetector
from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    DataFlowAnalysisService,
    RuntimeAnalysisService,
    DashboardMetricsService
)

# Configure logging for production (set to INFO to avoid debug messages)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class CodeAnalysisController:
    """
    Orchestrates the entire analysis: dynamic import, static analysis, token detection,
    data flow analysis, and runtime checks.
    """

    def __init__(self):
        self.global_namespace: Dict[str, Any] = {}
        self.runtime_analysis = RuntimeAnalysisService()

    def get_python_files(self, path: str) -> List[str]:
        """
        Recursively get Python files from a directory, or return the single file if a file is passed.
        """
        if os.path.isdir(path):
            python_files = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
            return python_files
        else:
            return [path]

    def start_analysis(self, path: str, threshold: float, num_tests: int) -> Dict[str, Any]:
        try:
            python_files = self.get_python_files(path)

            # 1️⃣ Dynamic Import & Populate Namespace
            self.global_namespace.clear()
            for file in python_files:
                module = DynamicImportService.dynamic_import(file)
                if module:
                    self.global_namespace.update(
                        {name: obj for name, obj in inspect.getmembers(module, inspect.isfunction)}
                    )

            # 2️⃣ Static Analysis
            static_analysis_result = StaticAnalysisService.parse_and_find_similarities(python_files, threshold)
            if not isinstance(static_analysis_result, tuple) or len(static_analysis_result) != 3:
                raise ValueError(f"Unexpected return from parse_and_find_similarities(): {static_analysis_result}")

            ast_trees, static_nodes, similarity_results = static_analysis_result

            # 3️⃣ Token-Based Clone Detection
            token_clones = [
                clone
                for file in python_files
                for clone in TokenBasedCloneDetector.detect_token_clones_with_ast(file, threshold)
            ]

            # 5️⃣ Data Flow Analysis
            data_flow_results = DataFlowAnalysisService.analyze_all_data_flow(ast_trees)

            # 6️⃣ Integrate Data Flow Dependencies
            for item in static_nodes + token_clones:
                for func_name in (item.get("func1"), item.get("func2")):
                    for filename, functions in data_flow_results.items():
                        if func_name in functions:
                            key = f"function{1 if func_name == item.get('func1') else 2}_metrics"
                            item.setdefault(key, {})["dependencies"] = functions[func_name].get("dependencies", {})

            for item in token_clones:
                f1_info, f2_info = item.get("function1_metrics", {}), item.get("function2_metrics", {})
                item["dataflow_similarity"] = (
                    compare_function_similarity(f1_info, f2_info)
                    if "dependencies" in f1_info and "dependencies" in f2_info
                    else 0.0
                )

            # 4️⃣ Generate Refactoring Suggestions
            refactoring_suggestions = RefactoringSug.generate_suggestions(token_clones, data_flow_results)

            # 7️⃣ Compute Dataflow Similarity
            for item in static_nodes + token_clones:
                f1_info, f2_info = item.get("function1_metrics", {}), item.get("function2_metrics", {})
                item["dataflow_similarity"] = (
                    compare_function_similarity(f1_info, f2_info)
                    if "dependencies" in f1_info and "dependencies" in f2_info
                    else 0.0
                )

            # 8️⃣ Runtime Analysis
            combined_results = static_nodes + token_clones
            valid_for_runtime = [r for r in combined_results if "function1_metrics" in r and "function2_metrics" in r]

            if valid_for_runtime:
                self.runtime_analysis.apply_runtime_checks(valid_for_runtime, self.global_namespace, num_tests)
                runtime_results = self.runtime_analysis.get_runtime_data()
            else:
                logging.warning("No valid function pairs for runtime analysis.")
                runtime_results = {}

            # 9️⃣ Compute Dashboard Metrics
            dashboard_metrics = DashboardMetricsService.compute_key_metrics(
                static_nodes, token_clones, runtime_results
            )

            # 🔟 Generate Analysis Insights
            analysis_insights = {
                "top_complex_functions": sorted(
                    [(item.get("function1_metrics", {}).get("name", ""),
                      item.get("function1_metrics", {}).get("complexity", 0))
                     for item in static_nodes] +
                    [(item.get("function2_metrics", {}).get("name", ""),
                      item.get("function2_metrics", {}).get("complexity", 0))
                     for item in static_nodes],
                    key=lambda x: x[1], reverse=True
                )[:3],
                "slowest_functions": sorted(
                    [(func.get("func_name", ""), max(iteration_data.get("total_wall_time", 0)
                                                     for iteration_data in func.get("test_results", {}).values()))
                     for func in runtime_results.get("functions", [])],
                    key=lambda x: x[1], reverse=True
                )[:3],
                "top_duplicate_pairs": sorted(
                    [(clone.get("func1", ""), clone.get("func2", ""), max(clone.get("token_similarity", 0.0),
                                                                          clone.get("ast_similarity", 0.0)))
                     for clone in token_clones] +
                    [(item.get("function1_metrics", {}).get("name", ""),
                      item.get("function2_metrics", {}).get("name", ""), item.get("similarity", 0.0))
                     for item in static_nodes],
                    key=lambda x: x[2], reverse=True
                )[:5],
                "files_with_most_duplicates": [
                    ("my_file1.py", 4),
                    ("my_file2.py", 2),
                ],
                "refactoring_suggestions": refactoring_suggestions,
                "key_metrics": dashboard_metrics,
            }

            # 🔚 Final Results
            return {
                "static": static_nodes,
                "token_clones": token_clones,
                "runtime": runtime_results,
                "data_flow": data_flow_results,
                "refactoring_suggestions": refactoring_suggestions,
                "analysis_insights": analysis_insights,
            }

        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Analysis failed: {e}\nTraceback:\n{tb_str}")
            raise RuntimeError(f"Error during analysis: {str(e)}\n\nTraceback:\n{tb_str}")
