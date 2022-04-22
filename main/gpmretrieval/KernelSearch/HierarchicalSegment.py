import gpbasics.global_parameters as global_param

global_param.ensure_init()

import tensorflow as tf
from gpbasics.Statistics.GaussianProcess import AbstractGaussianProcess
from gpbasics.KernelBasics import PartitioningModel as pm
from gpmretrieval.KernelSearch.Segment import Segment
from gpmretrieval.Partitioning import ChangePointModel as cpm


class HierarchicalSegment(Segment):
    """
    A hierarchical segment represents a certain fraction of given data (identified by a partitioning criterion) with a
    Gaussian process, which is comprised of a predefined mean function and a covariance function / kernel defined by
    this segment. It may have so-called descendants, which are hierarchical segments themselves. These descendant
    segments model a fraction of the data, which is modelled by their parent segment.
    """
    def __init__(self, gaussian_process: AbstractGaussianProcess, partition_criterion: pm.PartitionCriterion):
        super(HierarchicalSegment, self).__init__()
        self.partition_criterion: pm.PartitionCriterion = partition_criterion
        self.descendants: list = []
        self.latest_gp = gaussian_process

    def get_partition_criterion(self) -> pm.PartitionCriterion:
        return self.partition_criterion

    def add_descendant(self, partition):
        self.descendants.append(partition)

        if isinstance(self.partition_criterion, cpm.ChangePointCriterion) and self.partition_criterion is not None:
            cp_range_list = []
            if partition.partition_criterion.cp_range[0] < self.partition_criterion.cp_range[0]:
                cp_range_list.append(self.partition_criterion.cp_range[0])
            else:
                cp_range_list.append(partition.partition_criterion.cp_range[0])

            if partition.partition_criterion.cp_range[1] > self.partition_criterion.cp_range[1]:
                cp_range_list.append(self.partition_criterion.cp_range[1])
            else:
                cp_range_list.append(partition.partition_criterion.cp_range[1])

            partition.partition_criterion.cp_range = tf.concat(cp_range_list, axis=0)

    def get_descendants(self) -> list:
        return self.descendants