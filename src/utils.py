import os
import numpy as np
from itertools import zip_longest, accumulate, product,repeat
from types_utils import F
from typing import Any, Iterable, Callable, Sized
from threading import Thread


PATH = os.path.dirname(__file__)
RESOURCES_PATH = os.path.abspath(f"{PATH}/../resources")


def start_join_threads(threads: Iterable["Thread"]) -> None:
    """
    Starts all threads in threads and joins all the threads.
    :param threads: AN iterable of threads.
    """
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def grouper(iterable: Iterable, n: int):
    """
    Groups the iterable into a iterable of iterable of len n,
    e.g.((x0, x1, ..., xn-1), ((xn, xn+1, ..., x2n-1)), ...)
    :param iterable: The iterable to be grouped.
    :param n: The length of the groups. (The last group may be less the n in length.)
    """
    return zip_discard_generator(*([iter(iterable)] * n))


def zip_discard_generator(*iterables, sentinel: Any = object()):
    return ((entry for entry in iterable if entry is not sentinel)
            for iterable in zip_longest(*iterables, fillvalue=sentinel))


def parallel_evaluate_iterable(iterable, generate_thread_func: Callable[..., Thread], num_threads: int):
    """
    Evaluates a function over an iterable in parallel over several threads.
    :param iterable: The items to be evaluated.
    :param generate_thread_func: The function evaluating the items.
    :param num_threads: The number of threads to use.
    """
    if len(iterable) < num_threads:
        threads = map(generate_thread_func, iterable)
        start_join_threads(threads)
    else:
        for g in grouper(iterable, num_threads):
            threads = map(generate_thread_func, g)
            start_join_threads(threads)


def thread_wrapper(func: F):
    """
    Wraps a function into a thread call.
    :param func: THe function to be wrapped.
    """
    def wrapper(*args, **kwargs):
        return Thread(target=func, args=args, kwargs=kwargs)
    return wrapper


def partitions_with_overlap(image, partition_sizes, partitions_per_dim):
    """
    Partition an image with overlap to list of images.
    :param image: The image to partition.
    :param partition_sizes: The sizes of the partition in each dimension.
    :param partitions_per_dim: The number of partition per dimension.
    :return: A list of images.
    """
    shape = image.shape
    assert len(shape) == len(partition_sizes) == len(partitions_per_dim)

    dim_parts = []
    for s, p, n in zip(shape, partition_sizes, partitions_per_dim):
        strides = [(0, p)]
        if n > 1:
            overlap_diff = p - (p * n - s) / (n - 1)
            strides.extend([(a, a + p) for a in accumulate(repeat(overlap_diff, n - 1))])
        dim_parts.append(strides)

    return [image[[np.s_[round(d0):round(d1)] for d0, d1 in dim_splits]] for dim_splits in product(*dim_parts)]

