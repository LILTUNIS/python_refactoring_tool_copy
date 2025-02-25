import logging
import os
import traceback
from typing import Any, Dict, List

from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    TokenAnalysisService,
    DataFlowAnalysisService,
    RuntimeAnalysisService,
    DashboardMetricsService
)

# Configure logging as you wish
logging.basicConfig(
    level=logging.DEBUG,
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
            logging.debug(f"Starting analysis with {num_tests} test cases.")
            logging.debug(f"Threshold used = {threshold}")

            python_files = self.get_python_files(path)
            logging.debug(f"Number of Python files found = {len(python_files)}: {python_files}")

            # 1. Dynamic import & populate global namespace
            self.global_namespace.clear()
            for file in python_files:
                module = DynamicImportService.dynamic_import(file)
                if module:
                    import inspect
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        self.global_namespace[name] = obj
            logging.debug(f"Global namespace populated with {len(self.global_namespace)} functions.")

            # 2. Static Analysis
            ast_trees, static_nodes, similarity_results = \
                StaticAnalysisService.parse_and_find_similarities(python_files, threshold)

            logging.debug(f"static_nodes length = {len(static_nodes)}")
            logging.debug(f"similarity_results length = {len(similarity_results)}")

            # 3. Token-Based Analysis
            token_clones = TokenAnalysisService.detect_token_clones(python_files, threshold)
            logging.debug(f"token_clones length = {len(token_clones)}")

            # 4. Data Flow Analysis
            data_flow_results = DataFlowAnalysisService.analyze_all_data_flow(ast_trees)
            logging.debug(f"data_flow_results keys = {list(data_flow_results.keys())}")

            # 5. Runtime Analysis
            combined_results = static_nodes + token_clones
            logging.debug(f"combined_results length = {len(combined_results)}")

            if combined_results:
                valid_for_runtime = [
                    r for r in combined_results
                    if "function1_metrics" in r and "function2_metrics" in r
                ]
                logging.debug(f"valid_for_runtime length = {len(valid_for_runtime)}")
                logging.debug(f"valid_for_runtime = {valid_for_runtime}")

                if valid_for_runtime:
                    self.runtime_analysis.apply_runtime_checks(valid_for_runtime, self.global_namespace, num_tests)
                    runtime_results = self.runtime_analysis.get_runtime_data()
                else:
                    logging.warning("No valid function pairs found for runtime analysis.")
                    runtime_results = {}
            else:
                logging.warning("No function similarities detected; skipping runtime analysis.")
                runtime_results = {}

            # -----------------------------------------------------------
            # 6. Compute Key Metrics (DashboardMetricsService)
            # -----------------------------------------------------------
            # Use the previously fetched runtime results
            runtime_data = runtime_results
            logging.debug(f"[DEBUG] Runtime Data: {runtime_data}")

            # Pass runtime_data as the third argument
            dashboard_metrics = DashboardMetricsService.compute_key_metrics(static_nodes, token_clones, runtime_data)

            logging.debug(f"dashboard_metrics = {dashboard_metrics}")

            # -----------------------------------------------------------
            # 7. Compute Additional Dashboard Insights (Top Complex, Slowest, etc.)
            # -----------------------------------------------------------
            analysis_insights = {}

            # (A) Most Complex Functions
            complexity_data = []
            for item in static_nodes:
                f1 = item.get("function1_metrics", {})
                f2 = item.get("function2_metrics", {})
                if f1:
                    complexity_data.append((f1.get("name", ""), f1.get("complexity", 0)))
                if f2:
                    complexity_data.append((f2.get("name", ""), f2.get("complexity", 0)))
            complexity_data.sort(key=lambda x: x[1], reverse=True)
            analysis_insights["top_complex_functions"] = complexity_data[:3]
            logging.debug(f"[DEBUG] complexity_data (sorted) = {complexity_data}")

            # (B) Slowest Functions (by total wall time)
            slow_data = []
            funcs = runtime_results.get("functions", [])

            logging.debug(f"[DEBUG] 'runtime_results' -> {runtime_results}")

            for f in funcs:
                func_name = f.get("func_name", "")
                max_wall = 0
                test_results = f.get("test_results", {})

                logging.debug(f"[DEBUG] Analyzing '{func_name}', test_results={test_results}")

                for iteration_count, iteration_data in test_results.items():
                    # Show exactly what data we have for each iteration
                    logging.debug(f"  [DEBUG] iteration_count={iteration_count}, iteration_data={iteration_data}")

                    total_wall = iteration_data.get("total_wall_time", 0)
                    if total_wall > max_wall:
                        max_wall = total_wall

                slow_data.append((func_name, max_wall))

            slow_data.sort(key=lambda x: x[1], reverse=True)
            analysis_insights["slowest_functions"] = slow_data[:3]
            logging.debug(f"[DEBUG] Final slow_data: {slow_data}")

            # (C) Potential Code Duplication
            duplicates = []
            for clone in token_clones:
                f1, f2 = clone.get("func1", ""), clone.get("func2", "")
                token_sim = clone.get("token_similarity", 0.0)
                ast_sim = clone.get("ast_similarity", 0.0)
                best_sim = max(token_sim, ast_sim)
                duplicates.append((f1, f2, best_sim))
            # Also from static_nodes
            for item in static_nodes:
                f1 = item.get("function1_metrics", {}).get("name", "")
                f2 = item.get("function2_metrics", {}).get("name", "")
                sim = item.get("similarity", 0.0)
                duplicates.append((f1, f2, sim))

            duplicates.sort(key=lambda x: x[2], reverse=True)
            analysis_insights["top_duplicate_pairs"] = duplicates[:5]
            logging.debug(f"[DEBUG] top_duplicate_pairs: {analysis_insights['top_duplicate_pairs']}")

            # (D) Files with Most Duplicates (Placeholder)
            analysis_insights["files_with_most_duplicates"] = [
                ("my_file1.py", 4),
                ("my_file2.py", 2),
            ]

            # (E) Example: Simple Refactoring Suggestions
            refactoring_suggestions = []
            for func_name, complexity in complexity_data:
                if complexity > 10:
                    refactoring_suggestions.append(
                        f"Function '{func_name}' has high complexity ({complexity}); consider refactoring."
                    )
            if duplicates and duplicates[0][2] > 0.9:
                f1, f2, sim = duplicates[0]
                refactoring_suggestions.append(
                    f"Functions '{f1}' and '{f2}' have {sim * 100:.1f}% similarity; consider merging or refactoring."
                )
            analysis_insights["refactoring_suggestions"] = refactoring_suggestions

            # Attach the key metrics to the same dictionary
            analysis_insights["key_metrics"] = dashboard_metrics

            # -----------------------------------------------------------
            # Final Results
            # -----------------------------------------------------------
            final_results = {
                "static": static_nodes,
                "token_clones": token_clones,
                "runtime": runtime_results,
                "data_flow": data_flow_results,
                "analysis_insights": analysis_insights
            }

            logging.debug("Analysis completed successfully.")
            return final_results

        except Exception as e:
            tb_str = traceback.format_exc()
            logging.error(f"Analysis failed: {e}")
            logging.error(f"Traceback:\n{tb_str}")
            raise RuntimeError(f"Error during analysis: {str(e)}\n\nTraceback:\n{tb_str}")
