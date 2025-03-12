import importlib.util
import ast
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from model.static_model import static_model
from services.static_analyzer import (
    find_similar_nodes
)
from services.runtime_checker import RuntimeAnalyzer

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

        # Collect function metrics from static nodes (only if name is non-empty)
        for pair in static_nodes:
            for fn_key in ["function1_metrics", "function2_metrics"]:
                if fn := pair.get(fn_key, {}):
                    name = fn.get("name", "").strip()
                    if name:
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


def parse_files(python_files: List[str]) -> Dict[str, Any]:
    """
    Helper function that reads each Python file and parses it into an AST.
    Returns a dictionary mapping file paths to their corresponding AST.
    """
    ast_trees = {}
    for file in python_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file)
            ast_trees[file] = tree
        except Exception as e:
            logging.error(f"Failed to parse {file}: {e}")
    return ast_trees


class StaticAnalysisService:
    @staticmethod
    def parse_and_find_similarities(
        python_files: List[str],
        threshold: float
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[static_model], Dict[Tuple[str, str], Tuple[int, int]]]:
        # Parse all files into ASTs using the new helper function.
        ast_trees = parse_files(python_files)
        all_similar_nodes = []
        similarity_results = []

        # Dictionary to store each function’s line range:
        # key = (abs_file_path, function_name), value = (start_line, end_line)
        function_line_map = {}

        for file_path, tree in ast_trees.items():
            abs_file_path = os.path.abspath(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Find similar nodes using the imported function from static_analyzer.
            static_nodes = find_similar_nodes(
                tree,
                source_code,
                threshold,
                abs_file_path=abs_file_path
            )

            # Filter out any node pair with missing or empty function names.
            static_nodes = [
                node for node in static_nodes
                if node.get("function1_metrics", {}).get("name", "").strip() and
                   node.get("function2_metrics", {}).get("name", "").strip()
            ]

            # Collect each function’s start/end lines into function_line_map.
            for node_pair in static_nodes:
                func1_info = node_pair.get("function1_metrics", {})
                func2_info = node_pair.get("function2_metrics", {})

                func1_name = func1_info.get("name", "").strip()
                func2_name = func2_info.get("name", "").strip()

                start_line1 = func1_info.get("start_line")
                end_line1   = func1_info.get("end_line")
                if func1_name and start_line1 and end_line1:
                    function_line_map[(abs_file_path, func1_name)] = (start_line1, end_line1)

                start_line2 = func2_info.get("start_line")
                end_line2   = func2_info.get("end_line")
                if func2_name and start_line2 and end_line2:
                    function_line_map[(abs_file_path, func2_name)] = (start_line2, end_line2)

            all_similar_nodes.extend(static_nodes)
            # (Optionally, similarity_results can be built here if needed.)

        print("\n========== DEBUG: FULL STATIC ANALYSIS OUTPUT ==========")
        print(json.dumps(all_similar_nodes, indent=2))

        print("\n========== DEBUG: FULL TOKEN CLONES OUTPUT ==========")
        print(json.dumps(similarity_results, indent=2))

        return ast_trees, all_similar_nodes, similarity_results, function_line_map


class RuntimeAnalysisService:
    def __init__(self):
        self.runtime_analyzer = RuntimeAnalyzer()

    def apply_runtime_checks(self, similar_results, global_namespace, num_tests: int):
        self.runtime_analyzer.apply_runtime_checks(similar_results, global_namespace, num_tests)

    def get_runtime_data(self) -> Dict[str, Any]:
        runtime_data = self.runtime_analyzer.metrics
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
