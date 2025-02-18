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

# Set logging to INFO and disable DEBUG messages to reduce overhead during tests
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

    def update(self, execution_time: float, input_data: Dict[str, Any], output_data: Any) -> None:
        self.call_count += 1
        self.execution_time += execution_time
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
        """Decorator to wrap the function."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper

    def generate_dummy_arguments(self, func: Callable) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generate substantial test data with negative values,
        NO nested lists, and a fixed size (default: 10,000 elements) for scalability testing.
        """
        def generate_large_test_list(size=10000):
            return [random.randint(-100, 100) for _ in range(size)]

        signature = inspect.signature(func)
        args, kwargs = [], {}

        for name, param in signature.parameters.items():
            if param.default is not Parameter.empty:
                kwargs[name] = param.default
            elif param.annotation == int:
                args.append(42)
            elif param.annotation == float:
                args.append(3.14159)
            elif param.annotation == str:
                args.append("test_string")
            elif getattr(param.annotation, '__origin__', None) == list:
                # Use fixed-size large random list (default: 10,000 elements)
                args.append(generate_large_test_list(size=10000))
            elif getattr(param.annotation, '__origin__', None) == dict:
                kwargs[name] = {f"key_{i}": i for i in range(100)}
            elif param.kind == Parameter.VAR_POSITIONAL:
                args.extend(generate_large_test_list(size=10000))
            elif param.kind == Parameter.VAR_KEYWORD:
                kwargs.update({f"key_{i}": i for i in range(100)})
            else:
                args.append(generate_large_test_list(size=10000))

        return args, kwargs

    def _run_cpu_time_test(self, func: Callable, iterations: int,
                           args: List[Any], kwargs: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Measure precise CPU time using timeit for stable timings.
        A brief 0.01 second sleep is included after each call to reduce CPU spikes.
        """
        def test_func():
            result = func(*args, **kwargs)
            time.sleep(0.01)  # Reduce CPU spikes for more realistic CPU utilization
            if isinstance(result, (int, float, list, dict, str)):
                return hash(result)  # Prevent optimization
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
        Track current and peak memory usage,
        running gc.collect() before and after the measurement to minimize interference.
        """
        gc.collect()
        tracemalloc.clear_traces()
        tracemalloc.start()

        for _ in range(iterations):
            result = func(*args, **kwargs)
            if isinstance(result, (int, float)):
                pass

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()

        current_mem_kb = current_mem / 1024.0
        peak_mem_kb = peak_mem / 1024.0

        return current_mem_kb, peak_mem_kb

    def run_tests(self, num_tests: int) -> None:
        """
        Run dummy tests based on the number of test cases specified by the user.

        Args:
            num_tests (int): Number of test cases to generate.
        """
        logging.info(f"Generating {num_tests} test cases...")

        for _ in range(num_tests):
            # Simulating a dummy test execution
            time.sleep(random.uniform(0.01, 0.1))  # Simulate execution time

        logging.info("Tests completed.")
    def apply_runtime_checks(self, similar_nodes: List[Dict], global_namespace: Dict[str, Any], num_tests: int) -> None:
        """
        Apply runtime checks to provided functions using robust timing and memory profiling.

        Args:
            similar_nodes (List[Dict]): The detected function similarities.
            global_namespace (Dict[str, Any]): The namespace containing the functions.
            num_tests (int): The number of test iterations specified by the user.
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

                    args, kwargs = self.generate_dummy_arguments(original_func)

                    if func_name not in self.metrics:
                        self.metrics[func_name] = RuntimeMetrics()

                    # Run test exactly `num_tests` times
                    avg_cpu_time, total_wall_time, cpu_usage_percent = self._run_cpu_time_test(
                        decorated_func, num_tests, args, kwargs
                    )
                    total_memory_kb, peak_memory_kb = self._run_memory_test(
                        decorated_func, num_tests, args, kwargs
                    )

                    # Store the results
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
                        f"current_mem={total_memory_kb:.2f}KB, peak_mem={peak_memory_kb:.2f}KB"
                    )