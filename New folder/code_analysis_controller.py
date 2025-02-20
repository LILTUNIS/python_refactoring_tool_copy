import logging
import importlib.util
import os
import inspect
from operator import index
from typing import List, Dict, Any, Optional

from model.runtime_checker import RuntimeAnalyzer
from model.static_analyzer import SimilarityResult, parse_files, find_similar_nodes
from model.data_flow_analyzer import analyze_data_flow
from model.token_based_det import TokenBasedCloneDetector  # Import Token-Based Clone Detector

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class CodeAnalysisController:
    """
    Controller for performing code analysis combining static, token-based, runtime,
    and data flow analyses. It leverages AST parsing and dynamic module import to analyze
    Python code and produces actionable insights.
    """

    def __init__(self) -> None:
        """
        Initialize the controller with a runtime analyzer and a global namespace for imported functions.
        """
        self.global_namespace: Dict[str, Any] = {}  # To store dynamically imported functions
        self.similar_nodes: List[Dict[str, Any]] = []
        self.runtime_analyzer = RuntimeAnalyzer()

    @staticmethod
    def get_python_files(directory: str) -> List[str]:
        """
        Recursively retrieve all Python files in a given directory.

        Args:
            directory (str): The directory to search.

        Returns:
            List[str]: A list of paths to Python files.
        """
        python_files = []
        for root, _, files in os.walk(directory):
            python_files.extend([os.path.join(root, file) for file in files if file.endswith(".py")])
        return python_files

    def dynamic_import(self, file_path: str) -> Optional[Any]:
        """
        Dynamically import a Python module from the given file path.

        Args:
            file_path (str): The path to the Python file.

        Returns:
            module: The imported module if successful.

        Raises:
            RuntimeError: If the module cannot be imported.
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

    def start_analysis(self, path: str, threshold: float, num_tests: int) -> Dict[str, Any]:
        try:
            logging.debug(f"Starting analysis with {num_tests} test cases...")

            # Determine files to analyze
            python_files = self.get_python_files(path) if os.path.isdir(path) else [path]
            ast_trees = parse_files(python_files)

            all_similar_nodes = []
            all_data_flow_results = {}
            all_token_clones = []
            similarity_results = []  # Add this line to store SimilarityResult objects

            # --- Step 1: Dynamic Import & Global Namespace Population ---
            self.global_namespace = {}  # Reset namespace
            for file in python_files:
                module = self.dynamic_import(file)
                if module:
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        self.global_namespace[name] = obj
            logging.debug(f"Global namespace populated with {len(self.global_namespace)} functions.")

            # --- Step 2: Static & Token-Based Analysis, plus Data Flow ---
            for file, tree in ast_trees.items():
                logging.debug(f"Processing file: {file}")
                with open(file, "r", encoding="utf-8") as f:
                    source_code = f.read()

                    # AST-Based Static Analysis
                    static_nodes = find_similar_nodes(tree, source_code, threshold)
                    for node in static_nodes:
                        # Create SimilarityResult objects from static analysis
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
                    all_similar_nodes.extend(static_nodes)

                # Token-Based Clone Detection
                token_clones = TokenBasedCloneDetector.detect_token_clones_with_ast(file, threshold)
                all_token_clones.extend(token_clones)

                # Data Flow Analysis
                logging.debug(f"Analyzing data flow for {file}")
                # Data Flow Analysis
                data_flow_results = analyze_data_flow(tree)
                for func_name, func_data in data_flow_results.items():
                    all_data_flow_results[func_name] = func_data

            # --- Step 3: Runtime Analysis with User-Specified Test Cases ---
            if all_similar_nodes or all_token_clones:
                combined_results = all_similar_nodes + all_token_clones
                logging.debug(f"Combined Results for Runtime Analysis: {combined_results}")

                # Filter out any results that do not have the expected structure.
                valid_results = [res for res in combined_results if
                                 "function1_metrics" in res and "function2_metrics" in res]
                if not valid_results:
                    logging.warning("No valid function pairs found for runtime analysis after filtering.")
                    runtime_results = {}
                else:
                    self.runtime_analyzer.apply_runtime_checks(valid_results, self.global_namespace, num_tests)
                    runtime_results = self.get_runtime_data()
            else:
                logging.warning("Skipping runtime analysis - No function similarities detected.")
                runtime_results = {}

            # --- Step 4: Cross-Validation & Insights ---
            # Use only valid similar node entries (with the expected keys)
            valid_similar_nodes = [res for res in (all_similar_nodes + all_token_clones)
                                   if "function1_metrics" in res and "function2_metrics" in res]
            insights = self.cross_validate_results(valid_similar_nodes, runtime_results, all_data_flow_results)

            logging.debug("Analysis completed successfully.")
            return {
                "static": all_similar_nodes,
                "token_clones": all_token_clones,
                "runtime": runtime_results,
                "data_flow": all_data_flow_results,
                "insights": insights
            }
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise RuntimeError(f"Error during analysis: {str(e)}")

    def get_runtime_data(self) -> Dict[str, Any]:
        runtime_data = self.runtime_analyzer.get_metrics()
        functions = []
        total_memory_usage = 0.0
        peak_memory_usage = 0.0

        for func_name, metrics in runtime_data.items():
            memory_usage = metrics.memory_usage
            total_memory_usage += memory_usage
            peak_memory_usage = max(peak_memory_usage, memory_usage)

            function_metrics = {
                "func_name": func_name,
                "call_count": metrics.call_count,
                "execution_time": metrics.execution_time,
                "avg_time": metrics.avg_time,
                "memory_usage": memory_usage,
                "test_results": metrics.test_results,  # Added line!
                "input_patterns": list(metrics.input_patterns.keys()),
                "output_patterns": list(metrics.output_patterns.keys()),
            }
            functions.append(function_metrics)

        return {
            "functions": functions,
            "total_memory_usage": total_memory_usage,
            "peak_memory": peak_memory_usage,
        }

    def analyze_data_flow(self, file_path: str) -> Dict[str, Any]:
        """
        Perform data flow analysis on a single Python file.

        Args:
            file_path (str): The path to the file.

        Returns:
            dict: Data flow analysis results.

        Raises:
            ValueError: If the file cannot be parsed.
        """
        ast_trees = parse_files([file_path])
        if file_path in ast_trees:
            return analyze_data_flow(ast_trees[file_path])
        else:
            raise ValueError(f"Failed to parse file: {file_path}")

    def cross_validate_results(
            self,
            similar_nodes: List[Dict[str, Any]],
            metrics: Dict[str, Any],
            data_flow: Dict[str, Dict]
    ) -> List[str]:
        """
        Cross-validates similarity results, runtime metrics, and data flow analysis
        to generate actionable insights.

        Args:
            similar_nodes (List[Dict[str, Any]]): The results from static/token-based analyses.
            metrics (Dict[str, Any]): The runtime metrics.
            data_flow (Dict[str, Dict]): The data flow analysis results.

        Returns:
            List[str]: A list of insights/recommendations.
        """
        logging.debug("Starting cross_validate_results method.")

        if not isinstance(similar_nodes, list) or not similar_nodes:
            logging.error("similar_nodes is empty or not a valid list.")
            return []

        logging.debug(f"Full similar_nodes content: {similar_nodes}")
        insights = []

        def ensure_list(value: Any) -> List[Any]:
            if isinstance(value, list):
                return value
            elif isinstance(value, (str, int, float)):
                return [value]
            return []

        for index, result in enumerate(similar_nodes):
            logging.debug(f"Processing similar_nodes entry {index}: {result}")

            # Check that expected keys exist; if not, skip this entry.
            if "function1_metrics" not in result or "function2_metrics" not in result:
                logging.warning(
                    f"Skipping entry at index {index} due to missing function1_metrics/function2_metrics."
                )
                continue

            func1_metrics = result.get("function1_metrics")
            func2_metrics = result.get("function2_metrics")
            if not func1_metrics or not func2_metrics:
                logging.warning(
                    f"Skipping entry at index {index} due to empty function1_metrics/function2_metrics."
                )
                continue

            func1 = func1_metrics.get("name", "UnknownFunction1")
            func2 = func2_metrics.get("name", "UnknownFunction2")
            similarity = result.get("similarity", 0)
            token_similarity = result.get("token_similarity", 0)

            logging.debug(
                f"Comparing `{func1}` and `{func2}` with similarity scores - AST: {similarity:.2f}, Token: {token_similarity:.2f}"
            )

            data_flow1 = data_flow.get(func1)
            data_flow2 = data_flow.get(func2)

            if data_flow1 is None:
                logging.warning(f"Function `{func1}` not found in data_flow. Available keys: {list(data_flow.keys())}")
            if data_flow2 is None:
                logging.warning(f"Function `{func2}` not found in data_flow. Available keys: {list(data_flow.keys())}")

            reads1 = ensure_list(data_flow1.get("dependencies", {}).get("reads", [])) if data_flow1 else []
            reads2 = ensure_list(data_flow2.get("dependencies", {}).get("reads", [])) if data_flow2 else []
            writes1 = ensure_list(data_flow1.get("dependencies", {}).get("writes", [])) if data_flow1 else []
            writes2 = ensure_list(data_flow2.get("dependencies", {}).get("writes", [])) if data_flow2 else []

            logging.debug(f"Function `{func1}` - Reads: {reads1}, Writes: {writes1}")
            logging.debug(f"Function `{func2}` - Reads: {reads2}, Writes: {writes2}")

            deps1 = set(reads1)
            deps2 = set(reads2)
            shared_dependencies = deps1 & deps2
            consistent_dependencies = data_flow1 and data_flow2 and (
                data_flow1.get("dependencies") == data_flow2.get("dependencies")
            )

            logging.debug(f"Shared dependencies between `{func1}` and `{func2}`: {shared_dependencies}")
            logging.debug(f"Dependency structures match: {consistent_dependencies}")

            base_message = (
                f"Function Pair: `{func1}` and `{func2}`\n"
                f"  - AST Similarity: {similarity:.2f}\n"
                f"  - Token Similarity: {token_similarity:.2f}\n"
                f"  - Data Flow Dependencies Match: {consistent_dependencies}\n"
                f"  - Shared Dependencies: {shared_dependencies}\n"
            )
            insights.append(base_message)

        logging.debug(f"Finished processing cross_validate_results. Total insights generated: {len(insights)}")
        return insights


# Ensure logging is configured
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
