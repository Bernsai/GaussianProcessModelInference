from enum import Enum


# A Traversor is a required option for the IncrementalKernelSearch.
# A Traversor determines, when to proceed the kernel search for which segment. Each
# traversor has a different strategy to do that in the most efficient way. In doing so, each traversor provides a metric
# to sort the remaining, non-final segments, in order to determine, for which segment kernel search needs to be resumed.
class TraversorType(Enum):
    METRIC = 0
    LENGTH_WEIGHTED_METRIC = 1
    SEQUENTIAL = 2
    RANDOM = 3
