import time
from typing import Tuple
from statistics import mean, stdev


def timeit(num=1, eps=0.5, max_trial=5, warmup=0.01):
    """Create measurement function for excetution time

    Args:
        num (int): will execute the function as per the number is given here. (The default value is 1)
    """
    if num <= 0:
        raise ValueError(
            f"expected num is natural number. passed num={num} is out of range"
        )

    def _timeit(func, *args, **kwargs) -> Tuple[float, float]:
        """Return mean and std of execution time for func argument

        Args:
            func (Callable): function you want to execute
            args: arguments for function you want to execute
            kwargs: keyword arguments for function you want to execute

        Return:
            mean (float): the mean of execution time. the unit is milliseconds
            std (float): the standard deviation of execution time. the unit is milliseconds
        """
        for _ in range(max_trial):
            time.sleep(warmup)
            elapsed_times = []
            elapsed_a = elapsed_times.append
            for i in range(num):
                start = time.time()
                func(*args, **kwargs)
                elapsed_a(time.time() - start)
            mean_ = mean(elapsed_times) * 1000
            std_ = stdev(elapsed_times) * 1000
            if std_ / mean_ <= eps:
                break
        return mean_, std_

    return _timeit