from gpbasics import global_parameters as global_param

global_param.ensure_init()

import logging
import time
from gpmretrieval.Partitioning.ClusteringMethodType import ClusteringMethod
from multiprocessing.managers import SyncManager
from typing import List, Union
from gpmretrieval.ConcurrencyType import ConcurrencyType
import tensorflow as tf
import tensorflow_probability as tfp
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.PartitionOperator as po
import gpbasics.KernelBasics.PartitioningModel as pm
import gpmretrieval.ChangePointDetection.BatchCPDetection as bcpd
import gpmretrieval.KernelSearch.TraversorType as trty
import gpmretrieval.Partitioning.KmeansModel as pkm
import gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy as kexp
import gpmretrieval.KernelSearch.IncrementalTraversors as itr
import gpmretrieval.Partitioning.ChangePointModel as cpm
from gpbasics.KernelBasics import Kernel as k
from gpbasics.MeanFunctionBasics import MeanFunction as mf
from gpbasics.Metrics import Metrics as met
from gpbasics.Statistics import GaussianProcess as gp
from gpmretrieval.KernelSearch.IncrementalSegment import Partition
from gpmretrieval.KernelSearch.KernelSearch import KernelSearch, KernelSearchType
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpmretrieval.Partitioning.SklearnClusteringModel import SklearnPartitioningModel


def get_partitioning_stats(partitioned_data_input: di.PartitionedDataInput):
    block_sizes = []

    for idx, block_input in enumerate(partitioned_data_input.data_inputs):
        if isinstance(block_input, list):
            block_sizes.append(block_input[0].n_train)
        else:
            block_sizes.append(block_input.n_train)

    block_sizes = tf.constant(block_sizes, global_param.p_dtype)

    logging.debug("BlockwiseDataInput Stats: min %f, max %f, mean: %f, median %f"
                  % (tf.reduce_max(block_sizes), tf.reduce_min(block_sizes),
                     float(tf.reduce_mean(block_sizes)),
                     float(tfp.stats.percentile(block_sizes, 50.0, interpolation='midpoint'))))


class IncrementalKernelSearch(KernelSearch):
    def __init__(self, strategy_type: kexp.KernelExpansionStrategyType,
                 strategy_options: dict, data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]],
                 mean_function: mf.MeanFunction,
                 optimize_metric: met.MetricType, model_selection_metric: met.MetricType,
                 p_kernel_expression_replacement: bool,
                 cpd_type: bcpd.BatchChangePointDetectionType, clustering_type: ClusteringMethod,
                 global_approx: mht.GlobalApproximationsType, numerical_matrix_handling: mht.NumericalMatrixHandlingType,
                 covariance_function_repository: List[k.Kernel],
                 local_max_depth: int, random_restart: int = 1, subset_size: int = None,
                 incremental_traversor: trty.TraversorType = None,
                 ignored_dimensions: List[int] = None):
        if isinstance(data_input, list):
            raise Exception("IncrementalKernelSearch is currently not prepared for k-fold cross validation.")

        super(IncrementalKernelSearch, self).__init__(
            strategy_type, data_input, mean_function, KernelSearchType.Incremental, optimize_metric,
            model_selection_metric, p_kernel_expression_replacement, global_approx, numerical_matrix_handling,
            covariance_function_repository, random_restart, subset_size)

        self.strategy_options: dict = strategy_options

        self.traversor_type: trty.TraversorType = incremental_traversor

        # Initializing the employed traversing method
        if self.traversor_type is None or self.traversor_type is trty.TraversorType.SEQUENTIAL:
            self.traversor = itr.sequential_traversor
        elif self.traversor_type is trty.TraversorType.METRIC:
            self.traversor = itr.metric_traversor
        elif self.traversor_type is trty.TraversorType.LENGTH_WEIGHTED_METRIC:
            self.traversor = itr.length_weighted_metric_traversor
        elif self.traversor_type is trty.TraversorType.RANDOM:
            self.traversor = itr.random_traversor

        # Initializing the batch change point detection method, for partitioning uni-dimensional data
        self.cpd_type: bcpd.BatchChangePointDetectionType
        if cpd_type is None:
            self.cpd_type = bcpd.BatchChangePointDetectionType.BottomUp
        else:
            self.cpd_type = cpd_type

        # Initializing the clustering method, for partitioning multi-dimensional data
        self.clustering_type: ClusteringMethod
        if clustering_type is None:
            self.clustering_type = ClusteringMethod.KMEANS
        else:
            self.clustering_type = clustering_type

        # Concurrency Options set for IncrementalKernelSearch
        self.local_concurrency_type = ConcurrencyType.MULTI_THREADING
        self.activate_global_multi_threading = True
        self.local_max_depth: int = local_max_depth

        # Set, which dimensions should be ignored, when partitioning the given data.
        self.ignored_dimensions: List[int] = ignored_dimensions
        if self.ignored_dimensions is None:
            self.ignored_dimensions = []

    def get_clustering_model(self):
        """
        This method wraps the initialization of the clustering method, which is employed
        for partitioning multi-dimensional data
        Returns: a partitioning model

        """
        if self.clustering_type == ClusteringMethod.KMEANS:
            partitioning_model = pkm.KMeansModel(self.ignored_dimensions)
        elif self.clustering_type == ClusteringMethod.BIRCH:
            partitioning_model = SklearnPartitioningModel(self.ignored_dimensions, self.clustering_type)
        elif self.clustering_type == ClusteringMethod.GAUSSIAN_MIXTURE:
            partitioning_model = SklearnPartitioningModel(self.ignored_dimensions, self.clustering_type)
        else:
            raise Exception(f"Invalid clustering method '{self.clustering_type}'.")

        return partitioning_model

    def start_kernel_search(self) -> gp.AbstractGaussianProcess:
        partitioning_model: pm.PartitioningModel

        # Initializing the object, which performs the actual partitioning of the dataset
        # If the given dataset has uni-dimensional input data (X), change point detection is used
        if self.data_input.get_input_dimensionality() == 1:
            partitioning_model = cpm.ChangePointModel(self.cpd_type)

        # Otherwise, i.e. the given dataset has multi-dimensional input data (X), a clustering mechanism is used
        else:
            partitioning_model = self.get_clustering_model()

        # Find a partitioning scheme with the respective partitioning mechanism, which is
        # either change point detection or clustering.
        partitioning_model.automatic_init_criteria(self.data_input)

        logging.info("Initializing PartitionedDataInput.")
        start_time_bdi = time.time()

        # Partitioning the given data according to the found partitioning scheme
        mean_function = self.data_input.mean_function
        self.data_input: di.PartitionedDataInput = partitioning_model.partition_data_input(self.data_input)
        self.data_input.set_mean_function(mean_function)

        logging.info(f"Finished setup of PartitionedDataInput in {time.time() - start_time_bdi} seconds.")

        # IncrementalKernelSearch does not use global approximations for Gaussian processes. Therefore, cubic runtime
        # complexity of Gaussian processes may still negatively affect runtime, if the respective partitions are very
        # large. In order to identify such cases in hindsight and thus justify such long runtimes, statistics about the
        # partition sizes are logged, if DEBUG logging is activated.
        if global_param.p_logging_level == logging.DEBUG:
            get_partitioning_stats(self.data_input)

        activate_multiprocessing: bool = self.activate_global_multi_threading and global_param.p_max_threads > 1

        # If multiprocessing is used, auxiliary managers and locks need to be initialized.
        if activate_multiprocessing:
            logging.info("Initializing SyncManager.")
            SyncManager.register('Partition', Partition)
            manager: SyncManager = SyncManager()
            manager.start()
            self.orga_lock = manager.Lock()
            self.not_optimized = manager.Value('i', 0)

        segments: List[Partition] = []

        logging.info("Initializing list of segments.")

        # In order to manage kernel search for different partitions of the data, an auxiliary 'Partition'-object per
        # partition of the dataset is created. It holds information about the current status of the kernel search and
        # the current local model (also called local expert in the literature).
        for i in range(0, len(self.data_input.data_inputs)):
            criterion: pm.PartitionCriterion = partitioning_model.partitioning[i]

            max_n_train = self.data_input.data_inputs[i].n_train

            if max_n_train > 0:
                if activate_multiprocessing:
                    s: Partition = manager.Partition(
                        i, criterion, self.get_strategy_by_type(self.strategy_type, self.strategy_options))
                else:
                    s: Partition = Partition(
                        i, criterion, self.get_strategy_by_type(self.strategy_type, self.strategy_options))

                segments.append(s)

        logging.info(f"IncrementalKernelSearch with Traversor '{self.traversor_type}'.")

        # No Traversor specified or sequential traversal essentially means to
        # complete kernel search for every partition right away.
        if self.traversor_type is None or self.traversor_type == trty.TraversorType.SEQUENTIAL:
            logging.info(f"Starting search for kernels for {len(segments)} partition.")
            if activate_multiprocessing:
                global_param.pool.map(self.optimize_segment_completely, segments)
            else:
                for s in segments:
                    self.optimize_segment_completely(s)

        # If another traversor is specified, it determines, when to proceed the kernel search for which segment. Each
        # traversor has a different strategy to do that in the most efficient way.
        else:
            logging.info(f"Starting search for kernels for {len(segments)} partition.")
            filtered_segments: List[Partition] = segments
            while True:
                # All segments, whose local kernels / local Gaussian process models are considered final (no
                # further optimization possible), are thus not considered for further optimization. Therefore,
                # those are filtered from the respective list of partitions /segments.
                filtered_segments = list(filter(lambda segment: not segment.is_final(), filtered_segments))

                # Search is terminated, once every partition is final.
                if len(filtered_segments) == 0:
                    break

                # Those segments, which are already in the process of kernel search, are also not considered
                # for starting another kernel search, until the already running kernel search is finished.
                no_locked_filter: List[Partition] = \
                    list(filter(lambda segment: not segment.is_locked(), filtered_segments))

                # If for all non-final segments a kernel search is currently running, this outer procedure waits a
                # second before looking for optimizable segments again.
                if len(no_locked_filter) == 0:
                    if self.activate_global_multi_threading:
                        time.sleep(1)
                        continue

                # Select the segments, for which a kernel search should be initiated,
                # according to the specified traversor.
                worst_segments: List[Partition] = \
                    sorted(no_locked_filter, key=self.traversor, reverse=True)

                if len(worst_segments) > global_param.p_max_threads:
                    worst_segments = worst_segments[0:global_param.p_max_threads]

                # If multiprocessing is activated, a separate process is started for each kernel search per identified
                # segment.
                if activate_multiprocessing:
                    for ws in worst_segments:
                        ws.set_locked(True)

                    global_param.pool.map_async(self.optimize_segment, worst_segments)
                    logging.info(f"Started '{len(worst_segments)}' segment optimizations.")

                # Otherwise, a kernel search is started for the first element of the list of identified segments.
                else:
                    self.optimize_segment(worst_segments[0])

        logging.info("Building Divisive GPM from locally specialized sub-models (segment-wise models).")

        if activate_multiprocessing:
            global_param.shutdown_pool()

        # After kernel search has terminated for every segment (thus the local Gaussian process model for every segment
        # is final), a complete, divisive Gaussian process model for the whole dataset is built from the found
        # local models.
        result_gp: gp.AbstractGaussianProcess = \
            self.build_gaussian_process_from_segments(segments, partitioning_model, self.data_input)

        logging.info("Incremental Search ended.")

        if activate_multiprocessing:
            manager.shutdown()

        return result_gp

    def build_gaussian_process_from_segments(
            self, segments_list: List[Partition], partitioning_model: pm.PartitioningModel,
            data_input: di.PartitionedDataInput) -> gp.PartitionedGaussianProcess:
        """
        This method takes all the local Gaussian process models, which are assigned to the respective partitions, and
        builds one complete, divisive Gaussian process model for the whole dataset.
        Args:
            segments_list: list of segments / partitions, containing the respective local models.
            partitioning_model: the applied partitioning model
            data_input: the complete given data input

        Returns: Divisive Gaussian process model

        """
        child_nodes: List[k.Kernel] = [s.get_latest_gp().covariance_matrix.kernel for s in segments_list]

        partition_operator: po.PartitionOperator = po.PartitionOperator(
            self.data_input.get_input_dimensionality(), child_nodes, partitioning_model)

        omni_gp: gp.PartitionedGaussianProcess = gp.PartitionedGaussianProcess(partition_operator, self.mean_function)

        omni_gp.set_data_input(data_input)

        return omni_gp

    def optimize_segment(self, segment: Partition):
        """
        This method encapsulates the execution of one kernel search iteration for a given partition.
        Args:
            segment: the given partition / segment

        Returns:

        """
        block_data_input = self.data_input.data_inputs[segment.get_i()]
        strategy = segment.get_strategy()

        # execution of one kernel search iteration
        gaussian_process, metric, is_not_final = self.perform_kernel_search_iteration(
            segment.get_latest_gp(), segment.get_latest_metric(), block_data_input, strategy,
            self.local_concurrency_type, manager=None)

        segment.set_final(not is_not_final)
        segment.set_latest_metric(metric)
        segment.set_latest_gp(gaussian_process)
        segment.set_locked(False)

    def optimize_segment_completely(self, segment: Partition):
        """
        This method encapsulates the execution of a complete local kernel search for a given partition.
        Args:
            segment: the given partition / segment

        Returns:

        """
        block_data_input = self.data_input.data_inputs[segment.get_i()]
        segment.set_locked(True)
        strategy = segment.get_strategy()

        # execution of a complete local kernel search
        gaussian_process = self.perform_kernel_search_internal(strategy, self.local_concurrency_type, block_data_input)

        segment.set_final(True)
        segment.set_latest_metric(-1e+399)
        segment.set_latest_gp(gaussian_process)
        segment.set_locked(False)
