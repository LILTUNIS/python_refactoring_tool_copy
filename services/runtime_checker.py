# runtime_checker.py

import copy
import functools
import gc
import inspect
import logging
import os
import random
import time
import tracemalloc
from typing import Any, Dict, List, Tuple, Callable, get_origin, get_args

import coverage  # Used for coverage-based instrumentation to record executed lines

# Import RuntimeMetrics from the runtime model to store performance metrics.
from model.runtime_model import RuntimeMetrics
# Import function to extract usage information for function arguments.
from services.data_flow_analyzer import get_usage_info

# Configure logging to include timestamps, log level, and messages.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')


class RuntimeAnalyzer:
    """
    Analyzes runtime behavior for code clones using various metrics including:
      - Coverage (which lines and branches were executed)
      - Memory usage (via tracemalloc)
      - Timing information (CPU and wall time)

    This analyzer operates in a single process (no multiprocessing) and supports
    repeated tests for obtaining robust measurements.
    """

    def __init__(self, num_tests: int = 20, dummy_list_size: int = 100) -> None:
        # Dictionary mapping function names to their RuntimeMetrics.
        self.metrics: Dict[str, RuntimeMetrics] = {}
        self.num_tests = num_tests
        self.dummy_list_size = dummy_list_size
        # Frozen copy of metrics after runtime checks complete.
        self.frozen_metrics: Dict[str, RuntimeMetrics] = {}
        # Maps a tuple of (absolute file path, function name) to the function's start and end lines.
        self.function_line_map: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def runtime_check(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function in order to measure:
          - Execution time (per call)
          - Number of calls
          - Errors encountered during execution

        It increments the call count, captures execution time, and updates error counts.
        """
        if not func.__name__ or func.__name__.strip() == "":
            logging.warning("Skipping runtime check for unnamed function.")
            return func

        # Prevent decorating a function multiple times.
        if getattr(func, "_is_decorated", False):
            logging.debug(f"Function {func.__name__} already decorated; skipping duplicate.")
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()  # High-resolution wall clock start time.
            func_name = func.__name__
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # On exception, update error count for the function.
                if func_name not in self.metrics:
                    self.metrics[func_name] = RuntimeMetrics()
                self.metrics[func_name].error_count += 1
                logging.warning(f"Exception in function {func_name}: {e}", exc_info=True)
                return None
            end_ns = time.perf_counter_ns()  # End time measurement.

            if func_name not in self.metrics:
                self.metrics[func_name] = RuntimeMetrics()

            elapsed_sec = (end_ns - start_ns) / 1e9  # Convert nanoseconds to seconds.
            try:
                # Update metrics with elapsed time, passed arguments, and function result.
                self.metrics[func_name].update(
                    elapsed_sec,
                    {"args": args, "kwargs": kwargs},
                    result
                )
            except Exception as update_exception:
                logging.error(f"Error updating metrics for {func_name}: {update_exception}", exc_info=True)

            logging.debug(f"Decorator - {func_name}: Single-call time = {elapsed_sec:.6f}s")
            return result

        # Mark the function as decorated.
        wrapper._is_decorated = True
        return wrapper

    def generate_dummy_arguments(self, func: Callable) -> Tuple[list, dict]:
        """
        Generates dummy test arguments for a given function based on its signature and usage analysis.
        It uses default values when available or attempts to infer argument types (e.g., int, float, list, etc.)
        using both type annotations and usage data from get_usage_info.

        Returns:
            Tuple containing a list of positional arguments and a dictionary of keyword arguments.
        """
        if not func.__name__ or func.__name__.strip() == "":
            logging.warning("Function has no valid name. Returning empty dummy arguments.")
            return [], {}

        # Limits and ranges for dummy argument generation.
        RECURSIVE_LIMIT = 20
        LIST_SIZE_LIMIT = 5
        INT_RANGE = (-10, 10)

        def generate_test_list(size=LIST_SIZE_LIMIT, elem_type=int):
            # Generate a list of dummy elements based on the expected element type.
            if elem_type == int:
                return [random.randint(*INT_RANGE) for _ in range(size)]
            elif elem_type == float:
                return [random.uniform(-10, 10) for _ in range(size)]
            elif elem_type == str:
                return [f"string_{i}" for i in range(size)]
            elif elem_type == bool:
                return [random.choice([True, False]) for _ in range(size)]
            else:
                # Fallback for unsupported types.
                return [random.randint(*INT_RANGE) for _ in range(size)]

        # Get usage information (e.g., how arguments are used in the function).
        usage_data = get_usage_info(func)
        signature = inspect.signature(func)
        args, kwargs = [], {}

        for name, param in signature.parameters.items():
            # Use default value if one exists.
            if param.default != inspect.Parameter.empty:
                kwargs[name] = param.default
                continue

            # Check type annotation for type inference.
            origin = get_origin(param.annotation)
            args_info = get_args(param.annotation)

            if param.annotation == int:
                # Special case for known recursive functions.
                if func.__name__.lower() in ["fibonacci", "factorial"]:
                    args.append(random.randint(1, RECURSIVE_LIMIT))
                else:
                    args.append(random.randint(*INT_RANGE))

            elif param.annotation == float:
                args.append(random.uniform(-10, 10))

            elif param.annotation == str:
                args.append("test_string")

            elif param.annotation == bool:
                args.append(random.choice([True, False]))

            elif param.annotation == list or origin == list:
                # If element type is specified, generate a list of that type.
                if args_info:
                    elem_type = args_info[0]
                    args.append(generate_test_list(elem_type=elem_type))
                else:
                    args.append(generate_test_list())

            else:
                # Fallback to usage data inference if type annotation is absent or unrecognized.
                usage = usage_data.get(name, {})
                inferred_type = usage.get("type", "int")

                if inferred_type == "list":
                    val = generate_test_list()
                elif inferred_type == "float":
                    val = random.uniform(-10, 10)
                elif inferred_type == "str":
                    val = "inferred_string"
                elif inferred_type == "bool":
                    val = random.choice([True, False])
                elif inferred_type == "Callable":
                    val = lambda *a, **kw: None
                elif inferred_type == "dict":
                    val = {f"key_{i}": i for i in range(3)}
                else:
                    if usage.get("iterated") or usage.get("len_called"):
                        val = generate_test_list()
                    elif usage.get("arithmetic"):
                        val = random.randint(*INT_RANGE)
                    elif usage.get("bool_check"):
                        val = random.choice([True, False])
                    else:
                        val = random.randint(*INT_RANGE)
                args.append(val)

        return args, kwargs

    def _run_unified_test(self, func: Callable, func_name: str, iterations: int,
                          args: List[Any], kwargs: Dict[str, Any],
                          original_file_path: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        Runs a unified test for a function to measure:
          - CPU and wall time over a given number of iterations.
          - Memory usage via tracemalloc.
          - Code coverage information using the coverage module.

        Parameters:
            func: The function to test.
            func_name: Name of the function (used for logging and metrics).
            iterations: Number of times to run the function.
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            original_file_path: File path of the module containing the function.

        Returns:
            A tuple containing total CPU time, total wall time, and a dictionary with coverage and branch data.
        """
        abs_file_path = os.path.abspath(original_file_path)

        # Initialize coverage for the target file with branch tracking.
        cov = coverage.Coverage(branch=True, include=[abs_file_path])
        cov.erase()

        cov.start()
        tracemalloc.start()
        gc.collect()

        start_wall_ns = time.perf_counter_ns()
        start_cpu_ns = time.process_time_ns()

        for _ in range(iterations):
            gc.collect()  # Force garbage collection between iterations.
            snapshot_before = tracemalloc.take_snapshot()

            # Call the function directly.
            try:
                _ = func(*args, **kwargs)
            except Exception as e:
                # Log exception and update error count for this function.
                if func_name not in self.metrics:
                    self.metrics[func_name] = RuntimeMetrics()
                self.metrics[func_name].error_count += 1
                logging.warning(f"Exception in repeated calls of {func_name}: {e}")

            snapshot_after = tracemalloc.take_snapshot()
            # Compare memory snapshots to compute memory difference.
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            mem_diff = sum(abs(stat.size_diff) for stat in stats)
            # Update memory usage in KB.
            self.metrics[func_name].update_memory(mem_diff / 1024.0)

        end_wall_ns = time.perf_counter_ns()
        end_cpu_ns = time.process_time_ns()

        cov.stop()
        cov.save()
        tracemalloc.stop()

        # Compute total CPU and wall times in seconds.
        total_cpu_time = (end_cpu_ns - start_cpu_ns) / 1e9
        total_wall_time = (end_wall_ns - start_wall_ns) / 1e9

        cov_data = cov.get_data()
        coverage_map = {}
        branch_arcs_map = {}
        branch_execution_frequency = {}

        # Determine the function's code boundaries based on stored line numbers.
        start_line, end_line = self.function_line_map.get((abs_file_path, func_name), (0, 9999999))

        # Process coverage data for each measured file.
        for measured_file in cov_data.measured_files():
            executed_lines = cov_data.lines(measured_file) or []
            # Filter executed lines to only those within the function's line range.
            func_lines = [ln for ln in executed_lines if start_line <= ln <= end_line]
            coverage_map[measured_file] = sorted(func_lines)

            arcs = cov_data.arcs(measured_file)
            if arcs:
                filtered_arcs = [
                    (s_ln, e_ln)
                    for (s_ln, e_ln) in arcs
                    if (start_line <= s_ln <= end_line) or (start_line <= e_ln <= end_line)
                ]
                branch_arcs_map[measured_file] = sorted(filtered_arcs)

            executed_branch_arcs = cov_data.arcs(measured_file)
            if executed_branch_arcs:
                # Tally branch execution frequency for each arc.
                for (s_ln, e_ln) in executed_branch_arcs:
                    if ((start_line <= s_ln <= end_line) or (start_line <= e_ln <= end_line)):
                        if s_ln > 0 and e_ln > 0:
                            branch_execution_frequency[(s_ln, e_ln)] = \
                                branch_execution_frequency.get((s_ln, e_ln), 0) + 1

        branch_exec_count = sum(branch_execution_frequency.values())
        return total_cpu_time, total_wall_time, {
            "executed_lines": coverage_map,
            "branch_arcs": branch_arcs_map,
            "branch_execution_frequency": branch_execution_frequency,
            "branch_exec_count": branch_exec_count,
        }

    def apply_runtime_checks(self,
                             similar_nodes: List[Dict],
                             global_namespace: Dict[str, Any],
                             num_tests: int = None,
                             dummy_list_size: int = None) -> None:
        """
        Applies runtime checks on functions found in similar nodes.
        It decorates functions (if not already done) and then repeatedly
        calls them with generated dummy arguments to gather runtime metrics.

        Parameters:
            similar_nodes: A list of nodes containing runtime metrics for function pairs.
            global_namespace: Dictionary representing the global namespace where functions are defined.
            num_tests: (Optional) Number of times to test each function.
            dummy_list_size: (Optional) Size of dummy lists for test arguments.
        """
        if num_tests is None:
            num_tests = self.num_tests
        if dummy_list_size is None:
            dummy_list_size = self.dummy_list_size

        logging.debug(f"Entering apply_runtime_checks: num_tests={num_tests}, dummy_list_size={dummy_list_size}")
        decorated_funcs = set()  # Keep track of already decorated functions by (module, function name).

        # Iterate over similar nodes that contain metrics for two functions.
        for result in similar_nodes:
            func1 = result.get("function1_metrics")
            func2 = result.get("function2_metrics")
            if not func1 or not func2:
                logging.debug("Skipping a node because one of the function metrics is missing.")
                continue

            for fm in [func1, func2]:
                fn_name = fm.get("name", "").strip()
                file_path = fm.get("file_path", "")
                if not fn_name:
                    logging.warning("Skipping a function with missing name in runtime checks.")
                    continue

                logging.debug(f"Processing function: {fn_name}")
                # Check if the function exists in the provided global namespace.
                if fn_name in global_namespace:
                    original_func = global_namespace[fn_name]
                    key = (getattr(original_func, "__module__", ""), fn_name)
                    if key not in decorated_funcs:
                        decorated_funcs.add(key)
                        # Decorate the function to capture runtime metrics.
                        decorated_func = self.runtime_check(original_func)
                        global_namespace[fn_name] = decorated_func

                        logging.debug(f"Function {fn_name} decorated. New function id: {id(decorated_func)}")
                        if fn_name not in self.metrics:
                            self.metrics[fn_name] = RuntimeMetrics()
                            logging.debug(f"Metrics object created for function {fn_name}")

                        # Generate dummy arguments based on function signature and usage.
                        args, kwargs = self.generate_dummy_arguments(original_func)
                        logging.debug(f"Generated dummy arguments for {fn_name}: args={args}, kwargs={kwargs}")

                        total_cpu, total_wall, cov_info = self._run_unified_test(
                            decorated_func, fn_name, num_tests, args, kwargs, file_path
                        )
                        logging.debug(
                            f"Unified test complete for {fn_name}: "
                            f"total_cpu={total_cpu:.6f}s, total_wall={total_wall:.6f}s, cov_info={cov_info}"
                        )

                        coverage_map = cov_info.get("executed_lines", {})
                        branch_arcs_map = cov_info.get("branch_arcs", {})
                        branch_execution_frequency = cov_info.get("branch_execution_frequency", {})
                        branch_exec_count = sum(branch_execution_frequency.values())

                        calls = self.metrics[fn_name].call_count
                        # Update the runtime metrics for this function.
                        self.metrics[fn_name].test_results.setdefault(num_tests, {}).update({
                            "runtime": {
                                "avg_cpu_time": total_cpu / num_tests if num_tests else 0.0,
                                "total_wall_time": total_wall,
                                "avg_memory_kb": self.metrics[fn_name].avg_memory_kb,
                                "executed_lines": coverage_map,
                                "branch_arcs": branch_arcs_map,
                                "branch_execution_frequency": branch_execution_frequency,
                                "function_calls": calls,
                                "branch_executions": branch_exec_count,
                            }
                        })
                        logging.debug(
                            f"{fn_name} -> Calls: {calls}, "
                            f"Branches: {branch_exec_count}, "
                            f"Branch Exec Frequency: {branch_execution_frequency}, "
                            f"CPU Time: {total_cpu:.6f}s, Wall: {total_wall:.6f}s"
                        )
                    else:
                        logging.debug(f"Function {fn_name} already decorated; skipping duplicate decoration.")
                else:
                    logging.warning(f"Function {fn_name} not found in global_namespace; skipping runtime test.")

        # Freeze a deep copy of the final metrics for later retrieval or reporting.
        self.frozen_metrics = copy.deepcopy(self.metrics)
        logging.debug("apply_runtime_checks completed. Final frozen metrics: " + str(self.frozen_metrics))


#
# Optional similarity functions based on runtime metrics:
#

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Computes the Jaccard similarity between two sets.

    Returns 1.0 if both sets are empty, 0.0 if only one is empty, or the
    ratio of the intersection over the union otherwise.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_runtime_similarity(metrics1: RuntimeMetrics, metrics2: RuntimeMetrics,
                               w1: float = 0.4, w2: float = 0.3, w3: float = 0.2, w4: float = 0.1) -> float:
    """
    Computes a weighted similarity score between two functions based on their runtime metrics.

    The similarity score is a weighted sum of:
      - Time similarity (based on average execution time)
      - Memory similarity (based on average memory usage)
      - Branch coverage similarity (Jaccard similarity over executed branch arcs)
      - Input pattern similarity (Jaccard similarity over input patterns)

    Parameters:
        metrics1: RuntimeMetrics object for the first function.
        metrics2: RuntimeMetrics object for the second function.
        w1, w2, w3, w4: Weights for each component of the similarity score.

    Returns:
        A float value between 0.0 and 1.0 representing the overall similarity.
    """
    t1 = metrics1.avg_time
    t2 = metrics2.avg_time
    sim_time = 1 - abs(t1 - t2) / (max(t1, t2) + 1e-6)

    m1 = metrics1.avg_memory_kb
    m2 = metrics2.avg_memory_kb
    sim_memory = 1 - abs(m1 - m2) / (max(m1, m2) + 1e-6)

    def get_branch_execs(rm: RuntimeMetrics) -> set:
        # Extract branch execution frequency keys from the test results.
        if rm.test_results:
            first_key = next(iter(rm.test_results))
            freq = rm.test_results[first_key].get("runtime", {}).get("branch_execution_frequency", {})
            return set(freq.keys())
        return set()

    branches1 = get_branch_execs(metrics1)
    branches2 = get_branch_execs(metrics2)
    sim_branch = jaccard_similarity(branches1, branches2)

    calls1 = set(metrics1.input_patterns.keys())
    calls2 = set(metrics2.input_patterns.keys())
    sim_calls = jaccard_similarity(calls1, calls2)

    overall_sim = w1 * sim_time + w2 * sim_memory + w3 * sim_branch + w4 * sim_calls
    return max(0.0, min(overall_sim, 1.0))
