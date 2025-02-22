#analysis_services.py
import logging
import importlib.util
import os
from typing import Any, Dict, List, Optional

from model.runtime_checker import RuntimeAnalyzer
from model.static_analyzer import SimilarityResult, parse_files, find_similar_nodes
from model.data_flow_analyzer import analyze_data_flow
from model.token_based_det import TokenBasedCloneDetector

class DashboardMetricsService:
    @staticmethod
    def compute_key_metrics(
        static_nodes: List[Dict[str, Any]],
        token_clones: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute overall Key Metrics:
          - total_functions
          - total_static_pairs
          - avg_complexity
          - avg_loc
          - overall_similarity
          - total_duplicate_pairs
          ... etc.
        """

        # 1) Collect all unique functions from static_nodes
        #    (We see function1_metrics, function2_metrics for each "similar" pair)
        unique_function_names = set()
        complexities = []
        locs = []
        for pair in static_nodes:
            f1 = pair.get("function1_metrics", {})
            f2 = pair.get("function2_metrics", {})

            # Add function names
            if f1.get("name"):
                unique_function_names.add(f1["name"])
                complexities.append(f1.get("complexity", 0))
                locs.append(f1.get("loc", 0))

            if f2.get("name"):
                unique_function_names.add(f2["name"])
                complexities.append(f2.get("complexity", 0))
                locs.append(f2.get("loc", 0))

        # total_functions = how many distinct function names we found
        total_functions = len(unique_function_names)

        # total_static_pairs = just the length of static_nodes
        total_static_pairs = len(static_nodes)

        # avg_complexity
        avg_complexity = (sum(complexities) / len(complexities)) if complexities else 0.0

        # avg_loc
        avg_loc = (sum(locs) / len(locs)) if locs else 0.0

        # 2) Overall similarity
        #    We'll combine "token_clones" + "static_nodes" to compute an average
        #    or you might only want the "static_nodes" similarity, etc.
        #    For example, let's do an average of all similarities in static_nodes:
        similarities = [pair.get("similarity", 0.0) for pair in static_nodes]
        avg_ast_similarity = (sum(similarities) / len(similarities)) if similarities else 0.0

        #    Then also gather from token_clones (token_similarity + ast_similarity).
        #    If you want a single "overall" for token-based pairs, do (token + ast)/2, then average.
        token_asts = []
        for clone in token_clones:
            token_sim = clone.get("token_similarity", 0.0)
            ast_sim = clone.get("ast_similarity", 0.0)
            combined = (token_sim + ast_sim) / 2.0
            token_asts.append(combined)

        avg_token_ast = (sum(token_asts) / len(token_asts)) if token_asts else 0.0

        # Finally, let's define "overall_similarity" = average of these two means
        # (One approach: average the avg_ast_similarity with avg_token_ast)
        overall_similarity = (avg_ast_similarity + avg_token_ast) / 2.0 if (avg_ast_similarity or avg_token_ast) else 0.0

        # 3) total_duplicate_pairs
        #    For example, we might consider any pair with similarity > 0.8 as "duplicate."
        #    Or just count how many pairs are in static_nodes + token_clones.
        #    Let's do a simple approach: # of static_nodes + # of token_clones
        total_duplicate_pairs = len(static_nodes) + len(token_clones)

        return {
            "total_functions": total_functions,
            "total_static_pairs": total_static_pairs,
            "avg_complexity": avg_complexity,
            "avg_loc": avg_loc,
            "overall_similarity": overall_similarity,
            "total_duplicate_pairs": total_duplicate_pairs,
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

        for file, tree in ast_trees.items():
            with open(file, "r", encoding="utf-8") as f:
                source_code = f.read()
                static_nodes = find_similar_nodes(tree, source_code, threshold)
                all_similar_nodes.extend(static_nodes)

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
        """
        all_data_flow_results = {}
        for file_path, tree in ast_trees.items():
            logging.debug(f"Analyzing data flow for {file_path}")
            data_flow_results = analyze_data_flow(tree)
            for func_name, func_data in data_flow_results.items():
                key = f"{file_path}::{func_name}"
                all_data_flow_results[key] = func_data

        logging.debug("Data flow analysis complete.")
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
        total_memory_usage = 0.0
        peak_memory_usage = 0.0

        for func_name, metrics in runtime_data.items():
            memory_usage = metrics.memory_usage
            total_memory_usage += memory_usage
            peak_memory_usage = max(peak_memory_usage, memory_usage)

            functions.append({
                "func_name": func_name,
                "call_count": metrics.call_count,
                "execution_time": metrics.execution_time,
                "avg_time": metrics.avg_time,
                "memory_usage": memory_usage,
                "test_results": metrics.test_results,
                "input_patterns": list(metrics.input_patterns.keys()),
                "output_patterns": list(metrics.output_patterns.keys()),
            })

        return {
            "functions": functions,
            "total_memory_usage": total_memory_usage,
            "peak_memory": peak_memory_usage,
        }
