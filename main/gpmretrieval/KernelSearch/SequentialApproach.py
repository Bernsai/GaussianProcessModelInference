import gpbasics.global_parameters as global_param

global_param.ensure_init()

from gpmretrieval.ConcurrencyType import ConcurrencyType
import logging
from multiprocessing.managers import SyncManager
from typing import List, Tuple, cast, Union
import tensorflow as tf
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.Kernel as k
import gpmretrieval.ChangePointDetection.SequentialCPDetection as bscpd
from gpbasics.Statistics.GaussianProcess import AbstractGaussianProcess
import gpmretrieval.KernelSearch.Segment as seg
import gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy as kexp
from gpmretrieval.KernelSearch import KernelSearch as ks
from gpbasics.KernelBasics import Operators as op
from gpbasics.MeanFunctionBasics import MeanFunction as mf
from gpbasics.Metrics import Metrics as met
from gpbasics.Statistics import GaussianProcess as gp
from gpmretrieval.ChangePointDetection import SequentialCPDetection as scpd
from gpmretrieval.KernelSearch.KernelSearch import KernelSearchType
import gpbasics.Metrics.MatrixHandlingTypes as mht


class SequentialKernelSearch(ks.KernelSearch):
    def __init__(self, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]],
                 mean_function: mf.MeanFunction, default_window_size: int,
                 strategy_options: dict, kernelExpansion: kexp.KernelExpansionStrategyType,
                 sequential_cpd: scpd.SequentialChangePointDetectionType, optimize_metric: met.MetricType,
                 model_selection_metric: met.MetricType, p_kernel_expression_replacement: bool,
                 global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 covariance_function_repository: List[k.Kernel], random_restart: int = 1, subset_size: int = None):
        if isinstance(data_input, list):
            raise Exception("SequentialKernelSearch is currently not prepared for k-fold cross validation.")

        super(SequentialKernelSearch, self).__init__(
            kernelExpansion, data_input, mean_function, KernelSearchType.SKS, optimize_metric, model_selection_metric,
            p_kernel_expression_replacement, global_approx, numerical_matrix_handling, covariance_function_repository,
            random_restart, subset_size)

        self.default_window_size: int = default_window_size
        self.sequential_cpd_type: scpd.SequentialChangePointDetectionType = sequential_cpd
        self.sequential_cpd: scpd.SequentialChangePointDetection = self.get_cpd_by_type()
        self.strategy_options: dict = strategy_options
        self.activate_global_multi_threading: bool = True
        self.local_concurrency_type = ConcurrencyType.MULTI_THREADING

    def get_cpd_by_type(self) -> scpd.SequentialChangePointDetection:
        if self.sequential_cpd_type is scpd.SequentialChangePointDetectionType.WhiteNoise:
            return bscpd.WhiteNoiseCPD(self.data_input, self.default_window_size)
        else:
            logging.warning(
                "No valid SequentialChangePointDetectionType given (given: %s). "
                "Thus falling back to default type: WhiteNoise." % str(self.sequential_cpd_type))
            return bscpd.WhiteNoiseCPD(self.data_input, self.default_window_size)

    def start_kernel_search(self) -> gp.AbstractGaussianProcess:
        n_train = self.data_input.n_train
        x_train = self.data_input.data_x_train
        dim = self.data_input.get_input_dimensionality()

        change_points: List[tf.Tensor] = []

        # concurrency mechanisms can be used, if more than one thread / process can be used
        # and if global concurrency is activated
        concurrency: bool = global_param.p_max_threads > 1 and self.activate_global_multi_threading

        # If multiprocessing is used, auxiliary managers need to be initialized.
        if concurrency:
            manager: SyncManager = SyncManager()
            manager.start()
            segments_list: List[seg.LegacySegment] = manager.list()
        else:
            segments_list: List[seg.LegacySegment] = []

        # SKS / 3CS uses a window-based approach for identifying partitions in the specified data. This window is
        # defined via a start and an end index, which specify the window in accordance with the training data.
        # These indices combined with the training data are used to derive the actual change point positions.
        start_index: int = 0
        stop_index: int = min(n_train - 1, self.default_window_size)

        logging.info("Sequential Kernel Search started.")

        # If the available training data has more entries, than half of the used default window, it is assumed, that
        # finding change points makes sense in this scenario.
        if (self.default_window_size / 2) < n_train - 1:
            success: bool
            found_change_point: int
            while True:
                # Finding change points, and (if the first was successful) determine a Gaussian process model for
                # the resulting segment / partition of the data.
                success, found_change_point = self.usual_window_case(start_index, stop_index, segments_list)

                # if the previous step was not successful, the size of the considered window is enlarged.
                if not success:
                    if stop_index == n_train - 1:
                        break
                    stop_index = int(min([stop_index + self.default_window_size * 0.5, n_train - 1]))
                    continue

                # Otherwise, the found change point is added to the list of change points, and the indices defining the
                # window are reset to cover the next previously unconsidered area.
                else:
                    change_points.append(
                        cast(tf.Tensor, tf.Variable(
                            x_train[found_change_point], shape=[dim, ], dtype=global_param.p_dtype)))

                    start_index = found_change_point
                    stop_index = int(min([start_index + self.default_window_size, n_train - 1]))

                # If the last found change point is too close to the end of the dataset, the search for further change
                # points is terminated.
                if (start_index + self.default_window_size) > n_train + 0.25 * self.default_window_size:
                    break
        else:
            logging.info("No Segmentation done as default window size is larger than whole dataset.")

        # Usually, the last found change point (which was already used to initialize a new start index) is not directly
        # pointing to the end of the dataset. Thus, there is still one partition / segment, for which an apt Gaussian
        # process model needs to be found.
        if start_index < n_train - 1:
            stop_index = n_train
            if concurrency:
                self.start_find_kernel_for_segment_parallelized(start_index, stop_index, segments_list)
            else:
                segments_list.append(self.find_kernel_for_segment(start_index, stop_index))

        # Waiting for all concurrent processes to finish, before shutting down the pool
        if concurrency:
            global_param.shutdown_pool()

        # Build one Gaussian process from the local experts found per segment / partition
        omni_gp: gp.AbstractGaussianProcess = \
            self.build_gaussian_process_from_local_experts(segments_list, change_points)

        # Shutting down the synchronization manager for concurrent operations
        if concurrency:
            manager.shutdown()

        return omni_gp

    def build_gaussian_process_from_local_experts(
            self, segments_list: List[seg.LegacySegment], change_points: List[tf.Tensor]) -> AbstractGaussianProcess:
        """
        This method takes all the local Gaussian process models, which are assigned to the respective segments, and
        builds one complete, divisive Gaussian process model for the whole dataset.
        Args:
            segments_list: list of segments, which encapsulate the information about the found local models
            change_points: change points, which separate the given segments

        Returns: a Gaussian process model

        """
        if len(segments_list) > 1:
            # Important step as in multi_threading one segment calculation may overtake another one
            segments_list: List[seg.LegacySegment] = sorted(segments_list, key=lambda segment: segment.start_index)

            child_nodes: List[k.Kernel] = []
            for s in segments_list:
                child_nodes.append(s.latest_gp.covariance_matrix.kernel)

            dim = self.data_input.get_input_dimensionality()

            cp_operator_tree: op.ChangePointOperator = op.ChangePointOperator(
                dim, child_nodes, change_points)

            omni_gp: gp.BlockwiseGaussianProcess = gp.BlockwiseGaussianProcess(cp_operator_tree, self.mean_function)

            logging.debug("SKS built up full kernel expression from %i constituents." % len(omni_gp.constituent_gps))
        else:
            logging.warning("Resulting model has only one constituent segment.")
            omni_gp = segments_list[0].latest_gp

            logging.debug("SKS built up full kernel expression from 1 constituent.")

        return omni_gp

    def usual_window_case(self, start_index: int, stop_index: int, segments_list: List[seg.LegacySegment]) \
            -> Tuple[bool, int]:
        """
        This method combines the search for the next change point and (if the first has been found) a subsequent search
        for an apt Gaussian process model describing the resulting data partition.
        Args:
            start_index: defining the considered segment of the data, where a change point may be found.
            stop_index: defining the considered segment of the data, where a change point may be found.
            segments_list: the list of segments, to which a newly found Gaussian process model, respectively a
            Segment-object representing the latter, is added.

        Returns: a Tuple indicating (i) success of the change point detection, and (ii) the respective change point

        """
        success: bool
        new_change_point: int
        success, new_change_point = self.sequential_cpd.get_next_change_point(start_index, stop_index)

        if not success:
            return success, new_change_point
        else:
            logging.debug("Found new Change Point: %i" % new_change_point)

            if global_param.p_max_threads > 1 and self.activate_global_multi_threading:
                self.start_find_kernel_for_segment_parallelized(start_index, new_change_point, segments_list)
            else:
                segments_list.append(
                    self.find_kernel_for_segment(start_index, new_change_point))

            return success, new_change_point

    def find_kernel_for_segment(self, start_index: int, stop_index: int) -> seg.LegacySegment:
        """
        This method controls the search for a new Gaussian process model for a data segment specified by the window
        properties, i.e. start and end index. In doing so, this method extracts the respective data from the full
        data input and triggers a kernel search.
        Args:
            start_index: starting index of the found segment
            stop_index: last index of the found segment

        Returns:

        """
        logging.info(f"Find Kernel for Segment ({start_index} -- {stop_index}, #{stop_index - start_index} records).")

        # Indexing the training data for the data of the specified segment
        sliced_data_x_train = self.data_input.data_x_train[start_index:stop_index]
        sliced_data_y_train = self.data_input.data_y_train[start_index:stop_index]

        # Using the indexed training data to find the corresponding test data indices
        test_idx = tf.reshape(
            tf.where(tf.math.logical_and(
                tf.reshape(self.data_input.data_x_test, [-1, ]) < tf.reduce_max(sliced_data_x_train),
                tf.reshape(self.data_input.data_x_test, [-1, ]) >= tf.reduce_min(sliced_data_x_train))),
            [-1, ])

        # Using the test data indices to find the test data of the specified segment
        sliced_data_x_test = tf.gather(self.data_input.data_x_test, test_idx)
        sliced_data_y_test = tf.gather(self.data_input.data_y_test, test_idx)

        # Creating a DataInput-object for the new segment
        block_data_input = di.DataInput(
            sliced_data_x_train, sliced_data_y_train, sliced_data_x_test, sliced_data_y_test)
        block_data_input.set_mean_function(self.data_input.mean_function)

        # Initiate the kernel search
        gp_left = self.perform_kernel_search_internal(
            self.strategy_options, self.local_concurrency_type, block_data_input)

        # Saving the found information (i.e. Gaussian process model for the segment) to a Segment-object
        new_segment: seg.LegacySegment = seg.LegacySegment(start_index, stop_index)
        new_segment.set_latest_gp(gp_left)
        new_segment.set_final(True)

        return new_segment

    def start_find_kernel_for_segment_parallelized(
            self, start_index: int, stop_index: int, segments_list: List[seg.LegacySegment]):
        """
        This method starts 'find_kernel_for_segment' in a separate concurrent process.
        Args:
            start_index: starting index of the found segment
            stop_index: last index of the found segment
            segments_list: list of segments to append the newly found segment and its Gaussian process model to

        Returns:

        """

        global_param.pool.apply_async(self.find_kernel_for_segment_parallelizable,
                                      args=(start_index, stop_index, segments_list))

    def find_kernel_for_segment_parallelizable(
            self, start_index: int, stop_index: int, segments_list: List[seg.LegacySegment]):
        """
        This method wraps the functionality of the method 'find_kernel_for_segment' in a way, that it supports
        mechanisms for executing multiple instances of it concurrently. Therefore, a segment list is provided,
        which is synced across processes and threads.
        Args:
            start_index: starting index of the found segment
            stop_index: last index of the found segment
            segments_list: list of segments to append the newly found segment and its Gaussian process model to

        Returns:

        """
        new_segment: seg.LegacySegment = self.find_kernel_for_segment(start_index, stop_index)

        segments_list.append(new_segment)
