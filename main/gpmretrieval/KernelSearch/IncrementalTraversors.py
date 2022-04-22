from gpbasics import global_parameters as global_param

global_param.ensure_init()

import random
from gpmretrieval.KernelSearch import IncrementalSegment as iseg


# A Traversor is a required option for the IncrementalKernelSearch.
# A Traversor determines, when to proceed the kernel search for which segment. Each
# traversor has a different strategy to do that in the most efficient way. In doing so, each traversor provides a metric
# to sort the remaining, non-final segments, in order to determine, for which segment kernel search needs to be resumed.
def traversor_template(segment: iseg.Partition) -> float:
    pass


# Metric Traversor means, that the partition with the worst performance is taken next for resuming kernel search.
def metric_traversor(segment: iseg.Partition) -> float:
    return segment.get_latest_metric()


# Weighted Metric Traversor means, that the partition with the worst performance weighted by its size
# is taken next for resuming kernel search.
def length_weighted_metric_traversor(segment: iseg.Partition) -> float:
    return segment.get_latest_metric() * (segment.get_stop_index() - segment.get_start_index())


# A sequential traversor means, that the list of partitions is traversed by its initial order.
def sequential_traversor(segment: iseg.Partition) -> float:
    return segment.get_i()


# A random traversor picks a segment at random.
def random_traversor(segment: iseg.Partition) -> float:
    return random.random()
