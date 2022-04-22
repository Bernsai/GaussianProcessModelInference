from gpbasics import global_parameters as global_param

global_param.ensure_init()

from gpmretrieval.KernelExpansionStrategies import KernelExpansionStrategy as kexp
from gpmretrieval.KernelSearch.Segment import Segment
import gpbasics.KernelBasics.PartitioningModel as pm


class Partition(Segment):
    def __init__(self, i: int, partition_criterion: pm.PartitionCriterion, strategy: kexp.KernelExpansionStrategy):
        super(Partition, self).__init__()
        self.partition_criterion = partition_criterion
        self.locked = False
        self.final = False
        self.i: int = i
        self.strategy: kexp.KernelExpansionStrategy = strategy
        self.latest_metric: float = 1e+399

    def get_i(self):
        return self.i

    def get_partition_criterion(self) -> pm.PartitionCriterion:
        return self.partition_criterion

    def get_strategy(self):
        return self.strategy

    def set_strategy(self, strategy: kexp.KernelExpansionStrategy):
        self.strategy = strategy

    def is_locked(self):
        return self.locked

    def set_locked(self, locked):
        self.locked = locked

    def set_final(self, final):
        super(Partition, self).set_final(final)
