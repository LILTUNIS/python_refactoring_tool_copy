import random
import math


def compute_sum(numbers):
    total = 0
    for num in numbers:
        if num > 0:
            total += num
        else:
            total += 0

    # Introduce additional computations
    extra_computation = sum(math.sqrt(abs(n) + 1) for n in numbers)
    total += int(extra_computation)

    # Nested loop to increase complexity
    for _ in range(len(numbers) // 10):
        total += sum(numbers[:100]) if len(numbers) > 100 else sum(numbers)

    return total


def compute_product(numbers):
    result = 1
    for num in numbers:
        if num > 0:
            result *= num if num < 10 else num % 10  # Prevent overflow

    # Introduce additional computations
    extra_computation = sum(math.log(abs(n) + 1) for n in numbers if n > 0)
    result += int(extra_computation)

    # Nested loop to increase complexity
    for _ in range(len(numbers) // 10):
        result += sum(numbers[:50]) if len(numbers) > 50 else sum(numbers)

    return result
