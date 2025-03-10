import functools
import inspect
import os
import time
import tracemalloc
import random
import gc
import logging
import statistics
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Callable, get_origin, get_args

import coverage  # For coverage-based instrumentation

from model.data_flow_analyzer import get_usage_info

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

@dataclass
class RuntimeMetrics:
    """
    Stores runtime metrics for each function.
    """
    execution_time: float = 0.0
    execution_times: List[float] = field(default_factory=list)
    call_count: int = 0
    error_count: int = 0
    avg_time: float = 0.0
    memory_usages: List[float] = field(default_factory=list)
    avg_memory_kb: float = 0.0
    coverage_data: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    test_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # We'll store final function/branch counts here
    function_calls: int = 0
    branch_executions: int = 0

    # Tracks input / output patterns
    input_patterns: Dict[str, int] = field(default_factory=dict)
    output_patterns: Dict[str, int] = field(default_factory=dict)

    def update(self, elapsed: float, input_data: Dict[str, Any], output_data: Any) -> None:
        """
        Update aggregated metrics on each function call (basic timing) and input/output patterns.
        """
        self.call_count += 1
        self.execution_time += elapsed
        self.execution_times.append(elapsed)
        self.avg_time = self.execution_time / self.call_count

        input_key = str(input_data)
        self.input_patterns[input_key] = self.input_patterns.get(input_key, 0) + 1

        output_key = str(output_data)
        self.output_patterns[output_key] = self.output_patterns.get(output_key, 0) + 1

    def update_memory(self, mem_usage: float) -> None:
        """
        Update memory usage metrics.
        """
        self.memory_usages.append(mem_usage)
        self.avg_memory_kb = sum(self.memory_usages) / len(self.memory_usages) if self.memory_usages else 0.0


class RuntimeAnalyzer:
    """
    Analyzes runtime behavior for code clones.

    Key differences:
      - Uses a direct loop for each function instead of timeit.Timer.
      - Coverage, memory, and time are all measured together.
      - We freeze the metrics after tests to preserve them.
    """

    def __init__(self, num_tests: int = 20, dummy_list_size: int = 100) -> None:
        self.metrics: Dict[str, RuntimeMetrics] = {}
        self.num_tests = num_tests
        self.dummy_list_size = dummy_list_size
        self.frozen_metrics: Dict[str, RuntimeMetrics] = {}

        # This is set by the controller. Maps (abs_file_path, func_name) -> (start_line, end_line)
        self.function_line_map: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def runtime_check(self, func: Callable) -> Callable:
        """
        Decorator that increments call count, measures single-call execution time,
        and catches exceptions to update error counts.
        """
        if not func.__name__ or func.__name__.strip() == "":
            logging.warning("Skipping runtime check for unnamed function.")
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                if func_name not in self.metrics:
                    self.metrics[func_name] = RuntimeMetrics()
                self.metrics[func_name].error_count += 1
                logging.error(f"Exception in function {func_name}: {e}", exc_info=True)
                raise
            end_ns = time.perf_counter_ns()

            func_name = func.__name__
            if func_name not in self.metrics:
                self.metrics[func_name] = RuntimeMetrics()

            elapsed_sec = (end_ns - start_ns) / 1e9
            self.metrics[func_name].update(elapsed_sec, {"args": args, "kwargs": kwargs}, result)
            logging.debug(f"Decorator - {func_name}: Single-call time = {elapsed_sec:.6f}s")
            return result

        return wrapper

    def generate_dummy_arguments(self, func: Callable) -> Tuple[list, dict]:
        """
        Generates test.py arguments based on function signature and usage analysis.
        """
        if not func.__name__ or func.__name__.strip() == "":
            logging.warning("Function has no valid name. Returning empty dummy arguments.")
            return [], {}

        def generate_test_list(size=self.dummy_list_size, elem_type=int):
            if elem_type == int:
                return [random.randint(-100, 100) for _ in range(size)]
            elif elem_type == float:
                return [random.uniform(-100, 100) for _ in range(size)]
            elif elem_type == str:
                return [f"string_{i}" for i in range(size)]
            elif elem_type == bool:
                return [random.choice([True, False]) for _ in range(size)]
            else:
                return [random.randint(-100, 100) for _ in range(size)]

        usage_data = get_usage_info(func)
        signature = inspect.signature(func)
        args, kwargs = [], {}

        for name, param in signature.parameters.items():
            if param.default != inspect.Parameter.empty:
                kwargs[name] = param.default
                continue

            origin = get_origin(param.annotation)
            args_info = get_args(param.annotation)

            if param.annotation == int:
                args.append(random.randint(-100, 100))
            elif param.annotation == float:
                args.append(random.uniform(-100, 100))
            elif param.annotation == str:
                args.append("test_string")
            elif param.annotation == bool:
                args.append(random.choice([True, False]))
            elif (param.annotation == list or origin == list):
                if args_info:
                    elem_type = args_info[0]
                    args.append(generate_test_list(elem_type=elem_type))
                else:
                    args.append(generate_test_list())
            else:
                usage = usage_data.get(name, {})
                inferred_type = usage.get("type", "int")
                if inferred_type == "list":
                    val = generate_test_list()
                elif inferred_type == "float":
                    val = random.uniform(-100, 100)
                elif inferred_type == "str":
                    val = "inferred_string"
                elif inferred_type == "bool":
                    val = random.choice([True, False])
                elif inferred_type == "Callable":
                    val = lambda: None
                elif inferred_type == "dict":
                    val = {f"key_{i}": i for i in range(5)}
                else:
                    if usage.get("iterated") or usage.get("len_called"):
                        val = generate_test_list()
                    elif usage.get("arithmetic"):
                        val = random.randint(-100, 100)
                    elif usage.get("bool_check"):
                        val = random.choice([True, False])
                    else:
                        val = random.randint(-100, 100)
                args.append(val)

        return args, kwargs

    def _run_unified_test(
            self,
            func: Callable,
            func_name: str,
            iterations: int,
            args: List[Any],
            kwargs: Dict[str, Any],
            original_file_path: str
    ) -> Tuple[float, float, Dict[str, Any]]:

        abs_file_path = os.path.abspath(original_file_path)

        # Prepare coverage
        cov = coverage.Coverage(branch=True, include=[abs_file_path])
        cov.erase()
        cov.start()

        tracemalloc.start()
        gc.collect()

        start_wall_ns = time.perf_counter_ns()
        start_cpu_ns = time.process_time_ns()

        # Execute function multiple times
        for _ in range(iterations):
            gc.collect()
            snapshot_before = tracemalloc.take_snapshot()
            func(*args, **kwargs)
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            mem_diff = sum(abs(stat.size_diff) for stat in stats)
            self.metrics[func_name].update_memory(mem_diff / 1024.0)

        end_wall_ns = time.perf_counter_ns()
        end_cpu_ns = time.process_time_ns()

        cov.stop()
        cov.save()
        tracemalloc.stop()

        total_cpu_time = (end_cpu_ns - start_cpu_ns) / 1e9
        total_wall_time = (end_wall_ns - start_wall_ns) / 1e9

        # Collect coverage data
        cov_data = cov.get_data()
        coverage_map = {}
        branch_arcs_map = {}
        branch_execution_frequency = {}

        # Get (start_line, end_line) from function_line_map
        start_line, end_line = self.function_line_map.get((abs_file_path, func_name), (0, 9999999))

        for measured_file in cov_data.measured_files():
            executed_lines = cov_data.lines(measured_file) or []
            func_lines = [ln for ln in executed_lines if start_line <= ln <= end_line]
            coverage_map[measured_file] = sorted(func_lines)

            arcs = cov_data.arcs(measured_file)  # Get branch arcs
            if arcs:
                filtered_arcs = [
                    (s_ln, e_ln) for (s_ln, e_ln) in arcs
                    if (start_line <= s_ln <= end_line) or (start_line <= e_ln <= end_line)
                ]
                branch_arcs_map[measured_file] = sorted(filtered_arcs)

            # Extract branch execution frequency correctly
            executed_branch_arcs = cov_data.arcs(measured_file)
            if executed_branch_arcs:
                for (s_ln, e_ln) in executed_branch_arcs:
                    if (start_line <= s_ln <= end_line) or (start_line <= e_ln <= end_line):
                        if s_ln > 0 and e_ln > 0:  # ✅ Ensure only valid line numbers
                            branch_execution_frequency[(s_ln, e_ln)] = branch_execution_frequency.get((s_ln, e_ln), 0) + 1

        branch_exec_count = sum(branch_execution_frequency.values())  # Sum of all branch executions

        return total_cpu_time, total_wall_time, {
            "executed_lines": coverage_map,
            "branch_arcs": branch_arcs_map,
            "branch_execution_frequency": branch_execution_frequency,  # ✅ Fixed storage
            "branch_exec_count": branch_exec_count,  # ✅ Now correct
        }

    def apply_runtime_checks(
            self,
            similar_nodes: List[Dict],
            global_namespace: Dict[str, Any],
            num_tests: int = None,
            dummy_list_size: int = None
    ) -> None:
        """
        Apply runtime checks (coverage, memory, timing) to each function pair.
        Decorates each function once, then runs them in a loop to collect metrics.
        """
        if num_tests is None:
            num_tests = self.num_tests
        if dummy_list_size is None:
            dummy_list_size = self.dummy_list_size

        logging.debug(f"Entering apply_runtime_checks: num_tests={num_tests}, dummy_list_size={dummy_list_size}")
        decorated_funcs = set()  # ✅ Fixed

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

                if fn_name in global_namespace:
                    original_func = global_namespace[fn_name]
                    logging.debug(f"Original function {fn_name} found in global_namespace (id: {id(original_func)})")

                    # Only decorate once per function name
                    if fn_name not in decorated_funcs:
                        decorated_funcs.add(fn_name)
                        decorated_func = self.runtime_check(original_func)
                        global_namespace[fn_name] = decorated_func
                        logging.debug(f"Function {fn_name} decorated. New function id: {id(decorated_func)}")

                        # Ensure we have a metrics object
                        if fn_name not in self.metrics:
                            self.metrics[fn_name] = RuntimeMetrics()
                            logging.debug(f"Metrics object created for function {fn_name}")

                        # Generate dummy arguments
                        args, kwargs = self.generate_dummy_arguments(original_func)
                        logging.debug(f"Generated dummy arguments for {fn_name}: args={args}, kwargs={kwargs}")

                        # Run the unified test.py
                        total_cpu, total_wall, cov_info = self._run_unified_test(
                            decorated_func,
                            fn_name,
                            num_tests,
                            args,
                            kwargs,
                            file_path
                        )
                        logging.debug(
                            f"Unified test.py complete for {fn_name}: total_cpu={total_cpu:.6f}s, "
                            f"total_wall={total_wall:.6f}s, cov_info={cov_info}"
                        )

                        # Summarize coverage
                        coverage_map = cov_info.get("executed_lines", {})
                        branch_arcs_map = cov_info.get("branch_arcs", {})
                        branch_execution_frequency = cov_info.get("branch_execution_frequency", {})

                        # ✅ Fix branch execution count
                        branch_exec_count = sum(branch_execution_frequency.values())

                        # The call_count is updated each iteration, so read it now
                        calls = self.metrics[fn_name].call_count

                        # ✅ Prevent overwriting results
                        self.metrics[fn_name].test_results.setdefault(num_tests, {}).update({
                            "runtime": {
                                "avg_cpu_time": total_cpu / num_tests if num_tests else 0.0,
                                # ✅ Prevent ZeroDivisionError
                                "total_wall_time": total_wall,
                                "avg_memory_kb": self.metrics[fn_name].avg_memory_kb,
                                "executed_lines": coverage_map,
                                "branch_arcs": branch_arcs_map,
                                "branch_execution_frequency": branch_execution_frequency,  # ✅ Fixed storage
                                "function_calls": calls,
                                "branch_executions": branch_exec_count,  # ✅ Now correct
                            }
                        })

                        logging.debug(
                            f"{fn_name} -> Calls: {calls}, Branches: {branch_exec_count}, "
                            f"Branch Exec Frequency: {branch_execution_frequency}, "
                            f"CPU Time: {total_cpu:.6f}s, Wall: {total_wall:.6f}s"
                        )
                    else:
                        logging.debug(f"Function {fn_name} already decorated; skipping duplicate decoration.")
                else:
                    logging.warning(f"Function {fn_name} not found in global_namespace; skipping runtime test.py.")

        # Freeze metrics after all tests
        self.frozen_metrics = copy.deepcopy(self.metrics)
        logging.debug("apply_runtime_checks completed. Final frozen metrics: " + str(self.frozen_metrics))

    # --- New Helper Functions for Runtime Similarity Computation ---

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Compute the Jaccard similarity between two sets.
    Returns 1.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def compute_runtime_similarity(metrics1: RuntimeMetrics, metrics2: RuntimeMetrics,
                               w1: float = 0.4, w2: float = 0.3, w3: float = 0.2, w4: float = 0.1) -> float:
    """
    Compute a normalized runtime similarity score between 0 and 1 based on:
      - Sim_time: similarity in average execution time
      - Sim_memory: similarity in average memory usage (in KB)
      - Sim_branch: Jaccard similarity over executed branch arcs (keys of branch_execution_frequency)
      - Sim_calls: Jaccard similarity over input patterns (as a proxy for call patterns)
    The overall similarity is a weighted sum:
      Sim_runtime = w1*Sim_time + w2*Sim_memory + w3*Sim_branch + w4*Sim_calls
    """
    # Compute time similarity
    t1 = metrics1.avg_time
    t2 = metrics2.avg_time
    sim_time = 1 - abs(t1 - t2) / (max(t1, t2) + 1e-6)

    # Compute memory similarity
    m1 = metrics1.avg_memory_kb
    m2 = metrics2.avg_memory_kb
    sim_memory = 1 - abs(m1 - m2) / (max(m1, m2) + 1e-6)

    # Compute branch similarity (using keys from branch_execution_frequency)
    branches1 = set(metrics1.test_results.get(next(iter(metrics1.test_results), 0), {}).get("runtime", {}).get("branch_execution_frequency", {}).keys())
    branches2 = set(metrics2.test_results.get(next(iter(metrics2.test_results), 0), {}).get("runtime", {}).get("branch_execution_frequency", {}).keys())
    sim_branch = jaccard_similarity(branches1, branches2)

    # Compute call similarity using input patterns
    calls1 = set(metrics1.input_patterns.keys())
    calls2 = set(metrics2.input_patterns.keys())
    sim_calls = jaccard_similarity(calls1, calls2)

    overall_sim = w1 * sim_time + w2 * sim_memory + w3 * sim_branch + w4 * sim_calls
    # Ensure overall similarity is within [0, 1]
    overall_sim = max(0.0, min(overall_sim, 1.0))
    return overall_sim
