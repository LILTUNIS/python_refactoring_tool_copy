#code_analysis_controller.py
import logging
import os
from typing import Any, Dict, List

from services.analysis_services import (
    DynamicImportService,
    StaticAnalysisService,
    TokenAnalysisService,
    DataFlowAnalysisService,
    RuntimeAnalysisService
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
        """
        Main entry point for the entire analysis process.
        """
        try:
            logging.debug(f"Starting analysis with {num_tests} test cases.")
            python_files = self.get_python_files(path)

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

            # 3. Token-Based Analysis
            token_clones = TokenAnalysisService.detect_token_clones(python_files, threshold)

            # 4. Data Flow Analysis
            data_flow_results = DataFlowAnalysisService.analyze_all_data_flow(ast_trees)

            # 5. Runtime Analysis (only if we have any static or token-based findings)
            combined_results = static_nodes + token_clones
            if combined_results:
                valid_for_runtime = [
                    r for r in combined_results
                    if "function1_metrics" in r and "function2_metrics" in r
                ]
                if valid_for_runtime:
                    self.runtime_analysis.apply_runtime_checks(valid_for_runtime, self.global_namespace, num_tests)
                    runtime_results = self.runtime_analysis.get_runtime_data()
                else:
                    logging.warning("No valid function pairs found for runtime analysis.")
                    runtime_results = {}
            else:
                logging.warning("No function similarities detected; skipping runtime analysis.")
                runtime_results = {}

            logging.debug("Analysis completed successfully.")
            return {
                "static": static_nodes,
                "token_clones": token_clones,
                "runtime": runtime_results,
                "data_flow": data_flow_results
            }

        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise RuntimeError(f"Error during analysis: {str(e)}")
