#analysis_services.py
import logging
import importlib.util
import os
from typing import Any, Dict, List, Optional

from model.runtime_checker import RuntimeAnalyzer
from model.static_analyzer import SimilarityResult, parse_files, find_similar_nodes
from model.data_flow_analyzer import analyze_data_flow
from model.token_based_det import TokenBasedCloneDetector


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
