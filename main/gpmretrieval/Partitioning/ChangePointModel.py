import gpbasics.global_parameters as global_param

global_param.ensure_init()

from typing import List, cast
import logging
import time
import gpbasics.KernelBasics.PartitioningModel as pm
import gpbasics.DataHandling.DataInput as di
from gpmretrieval.ChangePointDetection import BatchCPDetection as bcpd
import tensorflow as tf
import tensorflow_probability as tfp


class ChangePointCriterion(pm.PartitionCriterion):
    def __init__(self, cp_range: List[tf.Tensor]):
        """
        Args:
            cp_range: cp_range[0] inclusive minimum value, cp_range[1] exclusive maximum value
        """
        super(ChangePointCriterion, self).__init__(pm.PartitioningClass.SELF_SUFFICIENT)
        assert len(cp_range) == 2, "Range per partitioning criterion declared insufficiently"
        self.cp_range = cp_range

    def get_score(self, x_vector: tf.Tensor) -> tf.Tensor:
        result_better = tf.logical_and(self.cp_range[0] <= tf.reshape(x_vector, [-1, ]),
                                       self.cp_range[1] > tf.reshape(x_vector, [-1, ]))
        return result_better

    def deepcopy(self):
        return ChangePointCriterion([r for r in self.cp_range])

    def get_json(self) -> dict:
        range_start = self.cp_range[0]
        if not isinstance(range_start, float):
            range_start = float(range_start)

        range_stop = self.cp_range[1]
        if not isinstance(range_stop, float):
            range_stop = float(range_stop)

        return {"type": "change_point", "start": range_start, "stop": range_stop}

    def __hash__(self):
        if isinstance(self.cp_range, tf.Tensor):
            return hash(tuple(self.cp_range.tolist()))

        return hash(tuple(self.cp_range))


class ChangePointModel(pm.PartitioningModel):
    def __init__(self, partitioning_type: bcpd.BatchChangePointDetectionType):
        super(ChangePointModel, self).__init__(pm.PartitioningClass.SELF_SUFFICIENT, [])
        self.partitioning_type = partitioning_type
        self.max_window_size = 500

    def automatic_init_criteria(self, data_input: di.DataInput, number_of_partitions: int = None,
                                predecessor_criterion: pm.PartitionCriterion = None):
        assert data_input.get_input_dimensionality() == 1, \
            "Change Point Model only works with uni-dimensional input data X"
        assert number_of_partitions is None or number_of_partitions >= 2

        # n number of partitions implies (n-1) change points
        if number_of_partitions is None:
            number_of_cps = None
        else:
            number_of_cps = number_of_partitions - 1

        if predecessor_criterion is not None:
            range = cast(ChangePointCriterion, predecessor_criterion).cp_range
            start = range[0]
            stop = range[1]
        else:
            start = -float("inf")
            stop = float("inf")

        cps = self.perform_initial_change_point_detection(data_input, number_of_change_points=number_of_cps)
        partitioning: List[pm.PartitionCriterion] = \
            self.get_partition_criteria_from_change_points(cps, start, stop, data_input.n_train)
        self.init_partitioning(partitioning)

    def perform_initial_change_point_detection(self, data_input: di.DataInput, number_of_change_points: int = None):
        cpd: bcpd.BatchChangePointDetection = self.get_change_point_detection_mechanism()

        logging.info("Starting change point detection by means of %s." % self.partitioning_type)

        start_time: float = time.time()
        if number_of_change_points is not None and number_of_change_points >= 1:
            cp_indices: List[int] = cpd.get_change_points(data_input, number_of_changepoints=number_of_change_points)
        else:
            cp_indices: List[int] = cpd.get_change_points(data_input)
        end_time: float = time.time()

        logging.info("%i Change points successfully retrieved in %f s." % (len(cp_indices), end_time - start_time))
        del start_time, end_time

        if global_param.p_logging_level is logging.DEBUG:
            cp_tf = tf.constant([0] + cp_indices + [data_input.n_train], dtype=global_param.p_dtype)
            distances: List[int] = cp_tf[1:] - cp_tf[:-1]
            logging.info(
                "Distances between change points: max: %i, min: %i, mean: %f, median: %i"
                % (tf.reduce_max(distances), tf.reduce_min(distances),
                   float(tf.reduce_mean(distances)),
                   float(tfp.stats.percentile(distances, 50.0, interpolation='midpoint'))))
            del distances

        cps: List[tf.Tensor] = []

        for c in cp_indices:
            cps.append(data_input.data_x_train[c])

        return cps

    def get_change_point_detection_mechanism(self) -> bcpd.BatchChangePointDetection:
        if self.partitioning_type is not None and self.partitioning_type.value > 100:
            return bcpd.BatchWrapperRuptures(self.partitioning_type)
        else:
            logging.warning("No BatchChangePointDetectionType given. Thus falling back to default type: Binary.")
            return bcpd.BatchWrapperRuptures(bcpd.BatchChangePointDetectionType.Binary)

    def get_partition_criteria_from_change_points(
            self, change_points: List[tf.Tensor], start: float = -float('inf'), stop: float = float('inf'), n: int = -1) \
            -> List[pm.PartitionCriterion]:
        result: List[pm.PartitionCriterion] = []

        if len(change_points) != 0:
            for i in range(len(change_points) + 1):
                cp_range: list
                cp_range_start: float
                cp_range_stop: float

                if i == 0:
                    cp_range_start = 0
                    cp_range_stop = float(change_points[i])
                elif i == len(change_points):
                    cp_range_start = float(change_points[i - 1])
                    cp_range_stop = 1
                else:
                    cp_range_start = float(change_points[i - 1])
                    cp_range_stop = float(change_points[i])

                result.append(ChangePointCriterion(tf.constant([cp_range_start, cp_range_stop], dtype=global_param.p_dtype)))
        else:
            result.append(ChangePointCriterion(tf.constant([-float("inf"), float("inf")], dtype=global_param.p_dtype)))
        return result

    def deepcopy(self):
        partitioning: List[ChangePointCriterion] = [pc.deepcopy() for pc in self.partitioning]
        copied_self = ChangePointModel(self.partitioning_type)
        copied_self.init_partitioning(partitioning)
        return copied_self
