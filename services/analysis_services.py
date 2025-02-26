#analysis_services.py
import logging
import importlib.util
import os
from typing import Any, Dict, List, Optional

from model.runtime_checker import RuntimeAnalyzer
from model.static_analyzer import SimilarityResult, parse_files, find_similar_nodes
from model.data_flow_analyzer import analyze_data_flow
from model.token_based_det import TokenBasedCloneDetector

import logging
logging.basicConfig(level=logging.DEBUG)

class DashboardMetricsService:
    @staticmethod
    def compute_key_metrics(
            static_nodes: List[Dict[str, Any]],
            token_clones: List[Dict[str, Any]],
            runtime_data: Dict[str, Any]  # New parameter for runtime metrics
    ) -> Dict[str, Any]:
        unique_function_names = set()
        complexities = []
        locs = []

        # Collect function metrics from static nodes
        for pair in static_nodes:
            for fn_key in ["function1_metrics", "function2_metrics"]:
                if fn := pair.get(fn_key, {}):
                    if name := fn.get("name", "").strip():  # Normalize names
                        unique_function_names.add(name)
                        complexities.append(fn.get("complexity", 0))
                        locs.append(fn.get("loc", 0))

        # Calculate basic metrics
        total_functions = len(unique_function_names)
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        avg_loc = sum(locs) / len(locs) if locs else 0.0

        pair_map = {}

        # Process AST similarities
        for pair in static_nodes:
            f1_name = pair.get("function1_metrics", {}).get("name", "").strip()
            f2_name = pair.get("function2_metrics", {}).get("name", "").strip()
            if f1_name and f2_name and f1_name != f2_name:
                key = tuple(sorted([f1_name, f2_name]))
                pair_map[key] = {
                    "ast_sim": pair.get("similarity", 0.0),
                    "token_sim": None
                }

        # Process Token similarities
        for clone in token_clones:
            f1_name = clone.get("function1_metrics", {}).get("name", "").strip()
            f2_name = clone.get("function2_metrics", {}).get("name", "").strip()
            if f1_name and f2_name and f1_name != f2_name:
                key = tuple(sorted([f1_name, f2_name]))
                token_sim = clone.get("token_similarity", 0.0)

                if key in pair_map:
                    # Merge token sim into existing AST entry
                    pair_map[key]["token_sim"] = max(
                        pair_map[key]["token_sim"] or 0,
                        token_sim
                    )
                else:
                    # Create new entry for token-based pair
                    pair_map[key] = {
                        "ast_sim": None,
                        "token_sim": token_sim
                    }

        # Calculate merged similarities
        all_similarities = []
        duplicate_pairs = 0

        for key, sims in pair_map.items():
            ast_val = sims["ast_sim"]
            token_val = sims["token_sim"]

            if ast_val is not None or token_val is not None:
                duplicate_pairs += 1

            if ast_val is not None and token_val is not None:
                all_similarities.append((ast_val + token_val) / 2)
            elif ast_val is not None:
                all_similarities.append(ast_val)
            else:
                all_similarities.append(token_val)

        overall_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0

        # === Calculate Memory Metrics ===
        total_avg_memory = 0.0
        peak_memory = 0.0

        # Sum up all avg memory values of all functions
        for func in runtime_data.get("functions", []):
            for test_data in func.get("test_results", {}).values():
                total_avg_memory += test_data.get("total_memory_kb", 0.0)
                peak_memory += test_data.get("peak_memory_kb", 0.0)

        # Debug prints (or replace with `print` if you prefer)
        print(f"[DEBUG] Total Avg Memory Dashboard: {total_avg_memory:.2f} KB")
        print(f"[DEBUG] Peak Memory Dashboard: {peak_memory:.2f} KB")

        # === Calculate Runtime Time Metrics (without 'most peak time') ===
        total_avg_time = sum(func.get("avg_time", 0.0) for func in runtime_data.get("functions", []))
        print(f"[DEBUG] Total Avg Time Dashboard: {total_avg_time:.6f} s")

        return {
            "total_functions": total_functions,
            "avg_complexity": round(avg_complexity, 2),
            "avg_loc": round(avg_loc, 2),
            "overall_similarity": round(overall_similarity, 2),
            "total_duplicate_pairs": duplicate_pairs,
            "total_avg_memory": round(total_avg_memory, 2),
            "peak_memory": round(peak_memory, 2),
            "total_avg_time": round(total_avg_time, 6),
        }




class DynamicImportService:
    @staticmethod
    def dynamic_import(file_path: str) -> Optional[Any]:
        """
        Dynamically import a Python module from the given file path.
        """
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to create module spec.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logging.debug(f"Module '{module_name}' successfully imported.")
            return module
        except Exception as e:
            logging.error(f"Error importing module '{file_path}': {e}")
            raise RuntimeError(f"Error importing module: {str(e)}")


class StaticAnalysisService:
    @staticmethod
    def parse_and_find_similarities(
        python_files: List[str],
        threshold: float
    ) -> (Dict[str, Any], List[Dict[str, Any]], List[SimilarityResult]):
        """
        Parse Python files into ASTs, then find similar nodes using static AST-based analysis.
        """
        ast_trees = parse_files(python_files)
        all_similar_nodes = []
        similarity_results = []

        # Debug: Initial State
        print("[DEBUG] Initial AST Trees:", ast_trees.keys())

        for file, tree in ast_trees.items():
            with open(file, "r", encoding="utf-8") as f:
                source_code = f.read()
                static_nodes = find_similar_nodes(tree, source_code, threshold)
                all_similar_nodes.extend(static_nodes)

                # Debug: Check Static Nodes
                print(f"[DEBUG] Static Nodes for file '{file}':", static_nodes)

                # Convert raw result dicts to typed SimilarityResult objects
                for node in static_nodes:
                    if "function1_metrics" in node and "function2_metrics" in node:
                        result = SimilarityResult(
                            node1=node["function1_metrics"].get("ast_pattern", ""),
                            node2=node["function2_metrics"].get("ast_pattern", ""),
                            similarity=node["similarity"],
                            complexity=node["function1_metrics"].get("complexity", 0),
                            return_behavior=node["function1_metrics"].get("return_behavior", ""),
                            loc=node["function1_metrics"].get("loc", 0),
                            parameter_count=node["function1_metrics"].get("parameter_count", 0),
                            nesting_depth=node["function1_metrics"].get("nesting_depth", 0),
                            ast_pattern=node["function1_metrics"].get("ast_pattern", {})
                        )
                        similarity_results.append(result)

                        # Debug: Check Similarity Result
                        print(f"[DEBUG] Similarity Result for file '{file}':", result)

        # Debug: Check Final Results Before Returning
        print("[DEBUG] Final AST Trees:", ast_trees)
        print("[DEBUG] Final All Similar Nodes:", all_similar_nodes)
        print("[DEBUG] Final Similarity Results:", similarity_results)
        print("[DEBUG] Returning:", (ast_trees, all_similar_nodes, similarity_results))

        logging.debug("Static analysis complete.")
        return ast_trees, all_similar_nodes, similarity_results


class TokenAnalysisService:
    @staticmethod
    def detect_token_clones(python_files: List[str], threshold: float) -> List[Dict[str, Any]]:
        """
        Perform token-based clone detection on a list of Python files.
        """
        all_token_clones = []
        for file in python_files:
            token_clones = TokenBasedCloneDetector.detect_token_clones_with_ast(file, threshold)
            all_token_clones.extend(token_clones)

        logging.debug("Token-based analysis complete.")
        return all_token_clones


class DataFlowAnalysisService:
    @staticmethod
    def analyze_all_data_flow(ast_trees: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform data flow analysis on all provided ASTs.

        Args:
            ast_trees (Dict[str, Any]): A dictionary mapping file paths to their corresponding ASTs.

        Returns:
            Dict[str, Any]: A dictionary containing data flow analysis results for each function.
                            Keys are formatted as "<file_path>::<function_name>".
        """
        all_data_flow_results = {}

        for file_path, tree in ast_trees.items():
            logging.info(f"[INFO] Starting data flow analysis for: {file_path}")

            try:
                # Perform data flow analysis on the AST
                data_flow_results = analyze_data_flow(tree)

                # Log a summary of the analysis
                logging.debug(
                    f"[DEBUG] Analysis complete for {file_path}. Functions analyzed: {list(data_flow_results.keys())}")

                # Store results with formatted keys
                for func_name, func_data in data_flow_results.items():
                    key = f"{file_path}::{func_name}"
                    all_data_flow_results[key] = func_data

            except Exception as e:
                logging.error(f"[ERROR] Data flow analysis failed for {file_path}: {str(e)}", exc_info=True)
                continue  # Continue with other files even if one fails

        logging.info("[INFO] Data flow analysis completed for all files.")
        return all_data_flow_results


class RuntimeAnalysisService:
    def __init__(self):
        self.runtime_analyzer = RuntimeAnalyzer()

    def apply_runtime_checks(self, similar_results, global_namespace, num_tests):
        """
        Apply runtime checks using the `RuntimeAnalyzer`.
        """
        print("[DEBUG] Applying runtime checks...")
        print(f"[DEBUG] Similar Results: {similar_results}")
        print(f"[DEBUG] Global Namespace: {list(global_namespace.keys())}")
        print(f"[DEBUG] Number of Tests: {num_tests}")

        self.runtime_analyzer.apply_runtime_checks(similar_results, global_namespace, num_tests)
        print("[DEBUG] Runtime checks applied.")

    def get_runtime_data(self) -> Dict[str, Any]:
        """
        Convert the raw runtime metrics into a dictionary for reporting.
        """
        print("[DEBUG] Getting runtime data...")
        runtime_data = self.runtime_analyzer.get_metrics()
        print(f"[DEBUG] Raw Runtime Data: {runtime_data}")

        functions = []
        peak_memory = 0.0

        # Iterate over each function's metrics
        for func_name, metrics in runtime_data.items():
            print(f"[DEBUG] Processing metrics for function: {func_name}")

            # Extract peak memory usage from test_results
            test_results = metrics.test_results
            for num_tests, result in test_results.items():
                peak_memory = max(peak_memory, result.get("peak_memory_kb", 0.0))

            # Append function-level details
            functions.append({
                "func_name": func_name,
                "call_count": metrics.call_count,
                "execution_time": metrics.execution_time,
                "avg_time": metrics.avg_time,
                "test_results": metrics.test_results,
                "input_patterns": list(metrics.input_patterns.keys()),
                "output_patterns": list(metrics.output_patterns.keys()),
            })

        print(f"[DEBUG] Peak Memory Calculated: {peak_memory:.2f} KB")
        print(f"[DEBUG] Final Functions List: {functions}")
        return {
            "functions": functions,
            "peak_memory": peak_memory,
        }
