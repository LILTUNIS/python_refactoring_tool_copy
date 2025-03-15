import importlib.util
import ast
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# Import static analysis model and service.
from model.static_model import static_model
from services.static_analyzer import find_similar_nodes
# Import runtime analysis service for dynamic performance checking.
from services.runtime_checker import RuntimeAnalyzer

# Configure logging at INFO level.
logging.basicConfig(level=logging.INFO)


class DynamicImportService:
    """
    Provides functionality to dynamically import a Python module given its file path.
    This service leverages importlib to load modules at runtime.
    """

    @staticmethod
    def dynamic_import(file_path: str) -> Optional[Any]:
        try:
            # Extract the module name from the file name.
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            # Create a module spec from the file location.
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to create module spec.")
            # Create a module based on the spec.
            module = importlib.util.module_from_spec(spec)
            # Execute the module to load its contents.
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logging.error(f"Error importing module '{file_path}': {e}")
            raise RuntimeError(f"Error importing module: {str(e)}")


def parse_files(python_files: List[str]) -> Dict[str, Any]:
    """
    Reads and parses each Python file into an Abstract Syntax Tree (AST).

    Parameters:
        python_files: List of file paths to Python source files.

    Returns:
        A dictionary mapping file paths to their corresponding AST.
    """
    ast_trees = {}
    for file in python_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                source = f.read()
            # Parse the source code into an AST.
            tree = ast.parse(source, filename=file)
            ast_trees[file] = tree
        except Exception as e:
            logging.error(f"Failed to parse {file}: {e}")
    return ast_trees


class StaticAnalysisService:
    """
    Provides services for performing static analysis on Python files.

    This includes:
      - Parsing files into ASTs.
      - Identifying similar code nodes (potential clones) using a similarity threshold.
      - Building a mapping of function line ranges for later use in runtime analysis.
    """

    @staticmethod
    def parse_and_find_similarities(
            python_files: List[str],
            threshold: float
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[static_model], Dict[Tuple[str, str], Tuple[int, int]]]:
        # Parse all provided files into their AST representations.
        ast_trees = parse_files(python_files)
        all_similar_nodes = []
        similarity_results = []  # Optionally, additional similarity data can be collected here.

        # Dictionary mapping (absolute file path, function name) to its (start_line, end_line).
        function_line_map = {}

        for file_path, tree in ast_trees.items():
            abs_file_path = os.path.abspath(file_path)

            # Read source code to support line-based operations.
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Use the static analyzer to find similar code nodes above the provided threshold.
            static_nodes = find_similar_nodes(
                tree,
                source_code,
                threshold,
                abs_file_path=abs_file_path
            )

            # Filter out node pairs with missing or empty function names.
            static_nodes = [
                node for node in static_nodes
                if node.get("function1_metrics", {}).get("name", "").strip() and
                   node.get("function2_metrics", {}).get("name", "").strip()
            ]

            # Collect start/end line ranges for each function encountered.
            for node_pair in static_nodes:
                func1_info = node_pair.get("function1_metrics", {})
                func2_info = node_pair.get("function2_metrics", {})

                func1_name = func1_info.get("name", "").strip()
                func2_name = func2_info.get("name", "").strip()

                start_line1 = func1_info.get("start_line")
                end_line1 = func1_info.get("end_line")
                if func1_name and start_line1 and end_line1:
                    function_line_map[(abs_file_path, func1_name)] = (start_line1, end_line1)

                start_line2 = func2_info.get("start_line")
                end_line2 = func2_info.get("end_line")
                if func2_name and start_line2 and end_line2:
                    function_line_map[(abs_file_path, func2_name)] = (start_line2, end_line2)

            # Aggregate all similar nodes for further analysis.
            all_similar_nodes.extend(static_nodes)
            # (Additional similarity_results could be computed and stored if needed.)

        # Print debugging output to aid in verification of analysis results.
        print("\n========== DEBUG: FULL STATIC ANALYSIS OUTPUT ==========")
        print(json.dumps(all_similar_nodes, indent=2))

        print("\n========== DEBUG: FULL TOKEN CLONES OUTPUT ==========")
        print(json.dumps(similarity_results, indent=2))

        return ast_trees, all_similar_nodes, similarity_results, function_line_map


class RuntimeAnalysisService:
    """
    Wraps the RuntimeAnalyzer service to apply runtime checks on analyzed functions.

    This service allows:
      - Applying runtime tests on functions that have been flagged as similar.
      - Retrieving collected runtime metrics for further analysis or reporting.
    """

    def __init__(self):
        self.runtime_analyzer = RuntimeAnalyzer()

    def apply_runtime_checks(self, similar_results, global_namespace, num_tests: int):
        """
        Applies runtime tests to functions identified in static analysis.

        Parameters:
            similar_results: List of similar node results (containing runtime metrics).
            global_namespace: Dictionary representing the global namespace with function definitions.
            num_tests: Number of tests to execute for each function.
        """
        self.runtime_analyzer.apply_runtime_checks(similar_results, global_namespace, num_tests)

    def get_runtime_data(self) -> Dict[str, Any]:
        """
        Gathers and returns runtime data from the RuntimeAnalyzer.

        This includes per-function metrics such as call counts, execution times,
        average times, test results, and input/output patterns. Also computes peak memory usage.

        Returns:
            A dictionary containing a list of function metrics and the overall peak memory usage.
        """
        runtime_data = self.runtime_analyzer.metrics
        functions = []
        peak_memory = 0.0
        # Iterate over each function's runtime metrics.
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
