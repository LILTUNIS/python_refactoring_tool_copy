import functools
import inspect
import time
import timeit
import tracemalloc
import gc
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Callable
from inspect import Parameter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.disable(logging.DEBUG)

@dataclass
class RuntimeMetrics:
    execution_time: float = 0.0
    call_count: int = 0
    avg_time: float = 0.0
    input_patterns: Dict[str, int] = field(default_factory=dict)
    output_patterns: Dict[str, int] = field(default_factory=dict)
    input_args: List[dict] = field(default_factory=list)
    memory_usage: float = 0.0
    total_memory: float = 0.0
    peak_memory: float = 0.0
    test_results: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def update(self, elapsed: float, input_data: Dict[str, Any], output_data: Any) -> None:
        self.call_count += 1
        self.execution_time += elapsed
        self.avg_time = self.execution_time / self.call_count

        input_key = str(input_data)
        self.input_patterns[input_key] = self.input_patterns.get(input_key, 0) + 1

        output_key = str(output_data)
        self.output_patterns[output_key] = self.output_patterns.get(output_key, 0) + 1


class RuntimeAnalyzer:
    def __init__(self) -> None:
        self.metrics: Dict[str, RuntimeMetrics] = {}

    def get_metrics(self) -> Dict[str, RuntimeMetrics]:
        return self.metrics

    def runtime_check(self, func: Callable) -> Callable:
        """Decorator that measures execution time on each call."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start

            func_name = func.__name__
            if func_name not in self.metrics:
                self.metrics[func_name] = RuntimeMetrics()

            input_data = {'args': args, 'kwargs': kwargs}
            self.metrics[func_name].update(elapsed, input_data, result)

            return result

        return wrapper

    def generate_dummy_arguments(self, func: Callable, list_size: int = 100) -> Tuple[List[Any], Dict[str, Any]]:
        """
        More robust dummy argument generator that:
        1. Respects known type hints (int, float, str, list, dict).
        2. Generates only one actual list if multiple list-typed parameters exist,
           to avoid "list * list" errors in user code that does direct multiplication.
        3. Falls back to int for untyped or extra list parameters.
        """

        def generate_test_list(size=list_size):
            return [random.randint(-100, 100) for _ in range(size)]

        signature = inspect.signature(func)
        args, kwargs = [], {}

        # Track how many list parameters we have assigned so far
        list_param_count = 0

        for name, param in signature.parameters.items():
            # 1. If there's a default, just use it.
            if param.default is not Parameter.empty:
                kwargs[name] = param.default
                continue

            # 2. Check explicit type hints
            if param.annotation == int:
                args.append(random.randint(-100, 100))
            elif param.annotation == float:
                args.append(random.uniform(-100, 100))
            elif param.annotation == str:
                args.append("test_string")
            elif getattr(param.annotation, '__origin__', None) == list:
                # If multiple list-typed params appear, only the first is a real list.
                # Additional list-typed params become integers to avoid "list * list" errors.
                list_param_count += 1
                if list_param_count == 1:
                    args.append(generate_test_list())
                else:
                    # Fallback to an integer for the second+ list param
                    args.append(random.randint(1, 10))
            elif getattr(param.annotation, '__origin__', None) == dict:
                kwargs[name] = {f"key_{i}": i for i in range(10)}

            # 3. Handle *args or **kwargs
            elif param.kind == Parameter.VAR_POSITIONAL:
                # We'll generate a single list for *args
                args.extend(generate_test_list())
            elif param.kind == Parameter.VAR_KEYWORD:
                # We'll generate a small dict for **kwargs
                kwargs.update({f"key_{i}": i for i in range(10)})

            # 4. If no annotation or unknown annotation, default to int
            else:
                args.append(random.randint(-100, 100))

        return args, kwargs

    def _run_cpu_time_test(self, func: Callable, iterations: int,
                           args: List[Any], kwargs: Dict[str, Any]) -> Tuple[float, float, float]:
        """Measure CPU and wall-clock time using timeit for stable timings."""

        def test_func():
            result = func(*args, **kwargs)
            # Avoid optimization by referencing result
            # but do not attempt to hash lists/dicts directly
            if isinstance(result, (int, float, str)):
                return hash(result)
            return 0

        timer = timeit.Timer(test_func)
        start_cpu_ns = time.process_time_ns()
        start_wall_ns = time.perf_counter_ns()
        total_wall_time = timer.timeit(number=iterations)
        end_wall_ns = time.perf_counter_ns()
        end_cpu_ns = time.process_time_ns()

        total_cpu_time = (end_cpu_ns - start_cpu_ns) / 1e9
        total_wall_time_measured = (end_wall_ns - start_wall_ns) / 1e9
        avg_cpu_time_per_call = total_cpu_time / iterations if iterations > 0 else 0.0
        cpu_usage_percent = (total_cpu_time / total_wall_time_measured) * 100 if total_wall_time_measured > 0 else 0.0

        return avg_cpu_time_per_call, total_wall_time_measured, cpu_usage_percent

    def _run_memory_test(self, func: Callable, iterations: int,
                         args: List[Any], kwargs: Dict[str, Any]) -> Tuple[float, float]:
        """
        Measure memory usage by taking tracemalloc snapshots before and after each call.
        Uses absolute differences to capture allocations or deallocations.
        """
        mem_diffs = []
        peak_mem_diffs = []
        tracemalloc.start()

        for _ in range(iterations):
            snapshot_before = tracemalloc.take_snapshot()
            func(*args, **kwargs)
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            # Sum absolute differences to capture net changes
            mem_diff = sum(abs(stat.size_diff) for stat in stats)
            mem_diffs.append(mem_diff)
            peak = max((abs(stat.size_diff) for stat in stats), default=0)
            peak_mem_diffs.append(peak)

        tracemalloc.stop()
        avg_mem_kb = (sum(mem_diffs) / len(mem_diffs)) / 1024.0 if mem_diffs else 0.0
        peak_mem_kb = (max(peak_mem_diffs) if peak_mem_diffs else 0) / 1024.0
        return avg_mem_kb, peak_mem_kb

    def run_tests(self, num_tests: int) -> None:
        """Run dummy tests based on the number of test cases specified by the user."""
        logging.info(f"Generating {num_tests} test cases...")

        for _ in range(num_tests):
            # Simulate a dummy test execution
            time.sleep(random.uniform(0.01, 0.1))  # Simulate execution time

        logging.info("Tests completed.")

    def apply_runtime_checks(self, similar_nodes: List[Dict], global_namespace: Dict[str, Any], num_tests: int,
                             dummy_list_size: int = 100) -> None:
        """
        Apply runtime checks to provided functions using robust timing and memory profiling.

        Args:
            similar_nodes (List[Dict]): The detected function similarities.
            global_namespace (Dict[str, Any]): The namespace containing the functions.
            num_tests (int): The number of test iterations.
            dummy_list_size (int): The size for generated list inputs (default: 100).
        """
        for result in similar_nodes:
            func1_metrics = result.get("function1_metrics")
            func2_metrics = result.get("function2_metrics")

            if not func1_metrics or not func2_metrics:
                continue

            for func_metrics in [func1_metrics, func2_metrics]:
                func_name = func_metrics.get("name")
                if func_name and func_name in global_namespace:
                    original_func = global_namespace[func_name]
                    decorated_func = self.runtime_check(original_func)
                    global_namespace[func_name] = decorated_func

                    # Generate robust dummy arguments
                    args, kwargs = self.generate_dummy_arguments(original_func, list_size=dummy_list_size)

                    if func_name not in self.metrics:
                        self.metrics[func_name] = RuntimeMetrics()

                    # Run tests exactly num_tests times and record CPU times
                    avg_cpu_time, total_wall_time, cpu_usage_percent = self._run_cpu_time_test(
                        decorated_func, num_tests, args, kwargs
                    )
                    total_memory_kb, peak_memory_kb = self._run_memory_test(
                        decorated_func, num_tests, args, kwargs
                    )

                    # Store the results in the metrics
                    self.metrics[func_name].test_results[num_tests] = {
                        "avg_cpu_time": avg_cpu_time,
                        "total_wall_time": total_wall_time,
                        "cpu_usage_percent": cpu_usage_percent,
                        "total_memory_kb": total_memory_kb,
                        "peak_memory_kb": peak_memory_kb,
                    }

                    print(
                        f"{func_name} [{num_tests}]: avg_cpu_time={avg_cpu_time:.6e}s, "
                        f"total_wall_time={total_wall_time:.6e}s, "
                        f"cpu_usage={cpu_usage_percent:.2f}%, "
                        f"avg_mem={total_memory_kb:.2f}KB, peak_mem={peak_memory_kb:.2f}KB"
                    )
