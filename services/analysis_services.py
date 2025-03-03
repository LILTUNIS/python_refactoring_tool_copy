import importlib.util
import os
from typing import Any, Dict, List, Optional

from model.runtime_checker import RuntimeAnalyzer
from model.static_analyzer import SimilarityResult, parse_files, find_similar_nodes
from model.data_flow_analyzer import analyze_data_flow

import logging
logging.basicConfig(level=logging.INFO)


class DashboardMetricsService:
    @staticmethod
    def compute_key_metrics(
            static_nodes: List[Dict[str, Any]],
            token_clones: List[Dict[str, Any]],
            runtime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        unique_function_names = set()
        complexities = []
        locs = []

        # Collect function metrics from static nodes
        for pair in static_nodes:
            for fn_key in ["function1_metrics", "function2_metrics"]:
                if fn := pair.get(fn_key, {}):
                    if name := fn.get("name", "").strip():
                        unique_function_names.add(name)
                        complexities.append(fn.get("complexity", 0))
                        locs.append(fn.get("loc", 0))

        total_functions = len(unique_function_names)
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0.0
        avg_loc = sum(locs) / len(locs) if locs else 0.0

        pair_map = {}

        # Combine AST, Token, and Dataflow Similarities
        for clone in token_clones:
            f1_name = clone.get("func1", "").strip()
            f2_name = clone.get("func2", "").strip()
            if f1_name and f2_name and f1_name != f2_name:
                key = tuple(sorted([f1_name, f2_name]))
                ast_sim = clone.get("ast_similarity")
                token_sim = clone.get("token_similarity")
                dataflow_sim = clone.get("dataflow_similarity")
                pair_map[key] = {
                    "ast_sim": ast_sim,
                    "token_sim": token_sim,
                    "dataflow_sim": dataflow_sim,
                }

        duplicate_pairs = 0
        token_similarities = []
        ast_similarities = []
        dataflow_similarities = []

        # Gather similarities from each pair
        for key, sims in pair_map.items():
            ast_val = sims.get("ast_sim")
            token_val = sims.get("token_sim")
            dataflow_val = sims.get("dataflow_sim")
            if any(val is not None for val in [ast_val, token_val, dataflow_val]):
                duplicate_pairs += 1
            if token_val is not None:
                token_similarities.append(token_val)
            if ast_val is not None:
                ast_similarities.append(ast_val)
            if dataflow_val is not None:
                dataflow_similarities.append(dataflow_val)

        token_avg = sum(token_similarities) / len(token_similarities) if token_similarities else None
        ast_avg = sum(ast_similarities) / len(ast_similarities) if ast_similarities else None
        dataflow_avg = sum(dataflow_similarities) / len(dataflow_similarities) if dataflow_similarities else None

        similarity_values = [val for val in [token_avg, ast_avg, dataflow_avg] if val is not None]
        overall_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else 0.0

        # Calculate Memory Metrics
        total_avg_memory = 0.0
        peak_memory = 0.0
        for func in runtime_data.get("functions", []):
            for test_data in func.get("test_results", {}).values():
                total_avg_memory += test_data.get("total_memory_kb", 0.0)
                peak_memory += test_data.get("peak_memory_kb", 0.0)

        # Calculate Runtime Time Metrics
        total_avg_time = sum(func.get("avg_time", 0.0) for func in runtime_data.get("functions", []))

        return {
            "total_functions": total_functions,
            "avg_complexity": round(avg_complexity, 2),
            "avg_loc": round(avg_loc, 2),
            "overall_similarity": round(overall_similarity, 2),
            "total_duplicate_pairs": duplicate_pairs,
            "total_avg_memory": round(total_avg_memory, 2),
            "peak_memory": round(peak_memory, 2),
            "total_avg_time": round(total_avg_time, 6),
            "dataflow_avg_similarity": round(dataflow_avg, 2) if dataflow_avg is not None else None,
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

        for file, tree in ast_trees.items():
            with open(file, "r", encoding="utf-8") as f:
                source_code = f.read()
                static_nodes = find_similar_nodes(tree, source_code, threshold)
                all_similar_nodes.extend(static_nodes)
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
                        # Optionally, add result to similarity_results if needed
                        # similarity_results.append(result)

        return ast_trees, all_similar_nodes, similarity_results


class DataFlowAnalysisService:
    @staticmethod
    def analyze_all_data_flow(ast_trees: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform data flow analysis on all provided ASTs.
        """
        all_data_flow_results = {}
        for file_path, tree in ast_trees.items():
            logging.info(f"[INFO] Starting data flow analysis for: {file_path}")
            try:
                data_flow_results = analyze_data_flow(tree)
                for func_name, func_data in data_flow_results.items():
                    key = f"{file_path}::{func_name}"
                    all_data_flow_results[key] = func_data
            except Exception as e:
                logging.error(f"[ERROR] Data flow analysis failed for {file_path}: {str(e)}", exc_info=True)
                continue
        logging.info("[INFO] Data flow analysis completed for all files.")
        return all_data_flow_results


class RuntimeAnalysisService:
    def __init__(self):
        self.runtime_analyzer = RuntimeAnalyzer()

    def apply_runtime_checks(self, similar_results, global_namespace, num_tests):
        """
        Apply runtime checks using the `RuntimeAnalyzer`.
        """
        self.runtime_analyzer.apply_runtime_checks(similar_results, global_namespace, num_tests)

    def get_runtime_data(self) -> Dict[str, Any]:
        """
        Convert the raw runtime metrics into a dictionary for reporting.
        """
        runtime_data = self.runtime_analyzer.get_metrics()
        functions = []
        peak_memory = 0.0
        for func_name, metrics in runtime_data.items():
            for num_tests, result in metrics.test_results.items():
                peak_memory = max(peak_memory, result.get("peak_memory_kb", 0.0))
            functions.append({
                "func_name": func_name,
                "call_count": metrics.call_count,
                "execution_time": metrics.execution_time,
                "avg_time": metrics.avg_time,
                "test_results": metrics.test_results,
                "input_patterns": list(metrics.input_patterns.keys()),
                "output_patterns": list(metrics.output_patterns.keys()),
            })
        return {
            "functions": functions,
            "peak_memory": peak_memory,
        }
