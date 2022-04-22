from gpbasics import global_parameters as global_param

global_param.ensure_init()

import logging
from typing import List, Union
import tensorflow as tf
from gpbasics.DataHandling.AbstractDataInput import AbstractDataInput
import gpmretrieval.autogpmr_parameters as agi_param
import gpbasics.DataHandling.DataInput as di
import gpbasics.KernelBasics.PartitionOperator as po
import gpbasics.KernelBasics.PartitioningModel as pm
import gpmretrieval.ChangePointDetection.BatchCPDetection as bcpd
import gpmretrieval.Partitioning.KmeansModel as pkm
import gpmretrieval.KernelExpansionStrategies.KernelExpansionStrategy as kexp
import gpmretrieval.Partitioning.ChangePointModel as cpm
import gpmretrieval.Partitioning.SklearnClusteringModel as skcm
from gpbasics.KernelBasics import Kernel as k
from gpbasics.Statistics.GaussianProcess import GaussianProcess, AbstractGaussianProcess
from gpbasics.MeanFunctionBasics import MeanFunction as mf
from gpbasics.Metrics import Metrics as met
from gpbasics.Statistics import GaussianProcess as gp
from gpmretrieval.KernelSearch.KernelSearch import KernelSearchType, KernelSearch
from gpmretrieval.Partitioning.ClusteringMethodType import ClusteringMethod
import gpbasics.Metrics.MatrixHandlingTypes as mht
from gpmretrieval.KernelSearch.HierarchicalSegment import HierarchicalSegment
from gpmretrieval.ConcurrencyType import ConcurrencyType
import ruptures as rpt


class TopDownHierarchicalApproach(KernelSearch):
    def __init__(
            self, strategy_type: kexp.KernelExpansionStrategyType, strategy_options: dict,
            data_input: Union[di.AbstractDataInput, List[di.AbstractDataInput]], mean_function: mf.MeanFunction,
            optimize_metric: met.MetricType, model_selection_metric: met.MetricType,
            p_kernel_expression_replacement: bool, cpd_type: bcpd.BatchChangePointDetectionType,
            global_approx: mht.GlobalApproximationsType, clustering_method: ClusteringMethod, local_max_depth: int,
            numerical_matrix_handling: mht.NumericalMatrixHandlingType,
            covariance_function_repository: List[k.Kernel], random_restart: int = 1, subset_size: int = None,
            ignored_dimensions: List[int] = None):
        if isinstance(data_input, list):
            raise Exception("TopDownHierarchicalApproach is currently not prepared for k-fold cross validation.")

        super(TopDownHierarchicalApproach, self).__init__(
            strategy_type, data_input, mean_function, KernelSearchType.TopDown_HKS, optimize_metric,
            model_selection_metric, p_kernel_expression_replacement, global_approx, numerical_matrix_handling,
            covariance_function_repository, random_restart, subset_size)

        self.strategy_options: dict = strategy_options
        self.cpd_type: bcpd.BatchChangePointDetectionType = cpd_type
        self.partitions_per_hierarchy_level: int = agi_param.p_default_partitions_split_per_layer
        self.local_max_depth: int = local_max_depth
        self.clustering_method: ClusteringMethod = clustering_method
        self.local_concurrency_type = ConcurrencyType.MULTI_PROCESSING

        # Although TopDownHierarchical Kernel search produces a Gaussian process model made up of local experts, its
        # hierarchical approach complicates global concurrency.
        self.activate_global_multi_threading: bool = False

        self.ignored_dimensions: List[int] = ignored_dimensions
        if self.ignored_dimensions is None:
            self.ignored_dimensions = []

    def start_kernel_search(self) -> gp.AbstractGaussianProcess:
        # TopDownHierarchicalApproach partitions the dataset in a hierarchical manner, i.e. it recursively (i) finds a
        # Gaussian process model for the given data, (ii) breaks down that data down into segments, and (iii) repeats
        # steps (i) and (ii) again. Thus one segment and its data may be further partitioned by descendant segments,
        # which also provide a more detailed explanation of the given data by means of an enhanced Gaussian process
        # model.

        # The hierarchical segment at the top of the lineage of segments represents the whole dataset, thus encompasses
        # no partition criterion.
        segment = HierarchicalSegment(None, None)

        layer_idx = 0

        # self.recursive_layer_processing performs the actual recursive kernel search procedure.
        self.recursive_layer_processing(
            segment, 1e+100, layer_idx, self.data_input)

        # The following lines of code extract the most detailed segments (highest layer idx).
        # at layer 0 of the hierarchy of segments (and the subsequent hierarchy of models) only one partition exists.
        segments_per_layer: List[List[HierarchicalSegment]] = [[segment]]

        for i in range(1, self.local_max_depth):
            segments_per_layer.append([])
            for segment in segments_per_layer[i - 1]:
                descendants = segment.get_descendants()
                if len(descendants) == 0:
                    segments_per_layer[i].append(segment)
                else:
                    segments_per_layer[i].extend(descendants)

        # Get the most detailed segments
        segment_set = segments_per_layer.pop()

        # Initialize the global partitioning model
        partitioning_model: pm.PartitioningModel = self.get_partitioning_method()

        partitioned_data_input: AbstractDataInput

        # If the most detailed partitioning only consists one partition /segment, no partitioned data input is required.
        if len(segment_set) == 1:
            partitioned_data_input = self.data_input

        # Otherwise, a partitioned data input is created.
        else:
            segment_criterion = [p.get_partition_criterion() for p in segment_set]

            partitioning_model.init_partitioning(segment_criterion)

            partitioned_data_input = partitioning_model.partition_data_input(
                self.data_input)

        # Building the resulting Gaussian process model corresponding to the found local experts
        # and the respective partitioning.
        result_gp = self.build_gaussian_process_from_segments(segment_set, partitioning_model, partitioned_data_input)

        # Shutting down the pool of processes for multiprocessing
        global_param.shutdown_pool()

        return result_gp

    def recursive_layer_processing(
            self, upper_layer_segment: HierarchicalSegment, upper_layer_metric: float,
            layer_idx: int, data_input: di.DataInput):
        """
        This method (i) finds a Gaussian process model for the given segment's data,
        and (ii) breaks down that data down into further segments. If the latter partitioning process is applicable,
        this method calls itself again for every single one of those segments.
        Args:
            upper_layer_segment: parent segment
            upper_layer_metric: quality of the parent segment's model with regards to the parent segment's data
            layer_idx: how many times this method has been called before
            data_input: segment's data

        Returns:

        """

        logging.info(f"Processing Data Input Partition of size train: {data_input.n_train}, test: {data_input.n_test}.")

        # Improve on the model of the parent segment by evaluating candidate kernels (and thus candidate Gaussian
        # process models) based on the parent model.
        best_gp: AbstractGaussianProcess
        layer_kernel_metric: float
        not_final: bool
        best_gp, layer_kernel_metric, not_final = \
            self.perform_kernel_search_iteration(
                upper_layer_segment.get_latest_gp(), upper_layer_metric, data_input,
                self.get_strategy_by_type(self.strategy_type, self.strategy_options), self.local_concurrency_type,
                manager=None)

        upper_layer_segment.set_latest_gp(best_gp)

        # granularity defines how many partitions ought to be found given the used subset size compared to the amount of
        # training data and the specified default granularity (which is self.partitions_per_hierarchy_level).
        granularity = int(
            min(tf.math.ceil(data_input.n_train / self.subset_size), self.partitions_per_hierarchy_level))

        # If at least 2 partitions should be found and a further extension of the current segment's model is possible
        # (i.e. not_final is True), the data of the given segment is further partitioned and
        if not_final and granularity >= 2 and self.local_max_depth > (layer_idx + 1):
            partitioning_model: pm.PartitioningModel

            if upper_layer_segment is not None:
                predecessor_criterion = upper_layer_segment.get_partition_criterion()
            else:
                predecessor_criterion = None

            partitioning_model: pm.PartitioningModel = self.get_partitioning_method()

            # Find a partitioning of the given segment's data
            try:
                partitioning_model.automatic_init_criteria(
                    data_input, number_of_partitions=granularity, predecessor_criterion=predecessor_criterion)
            except rpt.exceptions.BadSegmentationParameters as e:
                logging.info(
                    f"No more refinement possible due to {e} (partition size of {data_input.n_train} records).")
                return

            # Partition the given segment's data according to the found partitioning
            partitioned_data_input: di.PartitionedDataInput = partitioning_model.partition_data_input(data_input)

            # If at least one partition is too small in size, redo partitioning with reduced granularity.
            if min([pdi.n_train for pdi in partitioned_data_input.data_inputs]) < 50:
                # Only if the reduced granularity is still at least 2, find a new partitioning
                if (granularity - 1) >= 2:
                    granularity -= 1
                    partitioning_model.automatic_init_criteria(
                        data_input, number_of_partitions=granularity, predecessor_criterion=predecessor_criterion)

                    partitioned_data_input: di.PartitionedDataInput = \
                        partitioning_model.partition_data_input(data_input)

                # Otherwise, no further partitioning and kernel search is performed.
                else:
                    return

            partitioned_data_input.set_mean_function(self.mean_function)

            layer_idx += 1

            # Rerun this method for every resulting partition, in order to find more detailed local experts (which are
            # Gaussian process models here) for more fine-grained segments.
            for idx, data_input_part in enumerate(partitioned_data_input.data_inputs):
                partition = HierarchicalSegment(best_gp, partitioning_model.partitioning[idx])
                upper_layer_segment.add_descendant(partition)
                self.recursive_layer_processing(
                    partition, layer_kernel_metric, layer_idx, data_input_part)

        # Otherwise, no further partitioning and kernel search is performed.
        else:
            logging.info(f"No more refinement possible since (granularity < 2 or final) at layer idx {layer_idx} and "
                         f"a partition size of {data_input.n_train} records.")

    def get_partitioning_method(self) -> pm.PartitioningModel:
        """
        This method returns the partitioning method complying with the specified partitioning method types of this
        object.
        Returns: A Partitioning method (ChangePointModel for uni-dimensional input data and Clustering Model
        for multi-dimensional input data)

        """
        if self.data_input.get_input_dimensionality() == 1:
            partitioning_model = cpm.ChangePointModel(self.cpd_type)
        else:
            if self.clustering_method is ClusteringMethod.KMEANS:
                partitioning_model = pkm.KMeansModel(self.ignored_dimensions)
            else:
                partitioning_model = skcm.SklearnPartitioningModel(
                    self.ignored_dimensions, clustering_method=self.clustering_method)
        return partitioning_model

    def build_gaussian_process_from_segments(
            self, segments_list: List[HierarchicalSegment], partitioning_model: pm.PartitioningModel,
            data_input: di.DataInput) -> gp.AbstractGaussianProcess:
        """
        This method takes all the local Gaussian process models, which are assigned to the respective partitions, and
        builds one complete, divisive Gaussian process model for the whole dataset.
        Args:
            segments_list: list of segments / partitions, containing the respective local models.
            partitioning_model: the applied partitioning model
            data_input: the complete given data input

        Returns: Divisive Gaussian process model

        """
        child_nodes: List[k.Kernel] = []
        list_inducing_points: List[tf.Tensor] = []

        for idx, s in enumerate(segments_list):
            child_nodes.append(s.get_latest_gp().kernel)
            list_inducing_points.append(s.get_latest_gp().inducing_points)

        data_input.set_mean_function(self.mean_function)

        if len(child_nodes) == 1:
            omni_gp = gp.GaussianProcess(child_nodes[0], self.mean_function)
            omni_gp.set_inducing_points(inducing_points=list_inducing_points[0])
        else:
            cp_operator_tree: po.PartitionOperator = po.PartitionOperator(
                self.data_input.get_input_dimensionality(), child_nodes, partitioning_model)

            omni_gp = gp.PartitionedGaussianProcess(cp_operator_tree, self.mean_function)

            omni_gp.set_inducing_points(list_inducing_points)

        omni_gp.set_data_input(data_input)

        return omni_gp
