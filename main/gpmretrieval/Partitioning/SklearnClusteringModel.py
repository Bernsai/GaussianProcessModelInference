import gpbasics.global_parameters as global_param

global_param.ensure_init()

from sklearn.cluster import Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from typing import List
import gpbasics.KernelBasics.PartitioningModel as pm
import gpbasics.DataHandling.DataInput as di
from gpmretrieval.Partitioning.ClusteringMethodType import ClusteringMethod
import time
import logging
import tensorflow as tf


class SklearnClusterCriterion(pm.PartitionCriterion):
    def __init__(self, model, cluster_id, predecessor: pm.PartitionCriterion, clustering_method: ClusteringMethod):
        super(SklearnClusterCriterion, self).__init__(pm.PartitioningClass.SELF_SUFFICIENT)
        self.model = model
        self.cluster_id = cluster_id
        self.predecessor: pm.PartitionCriterion = predecessor
        self.clustering_method = clustering_method

    def get_score(self, x_vector: tf.Tensor) -> tf.Tensor:
        if self.predecessor is not None:
            pre_score = self.predecessor.get_score(x_vector)
            if tf.reduce_sum(pre_score) == 0:
                return pre_score
        else:
            pre_score = tf.ones(shape=[x_vector.shape[0], ], dtype=global_param.p_dtype)

        fit_predict = tf.reshape(self.model.predict(x_vector), [-1, ])
        score = tf.cast(fit_predict == self.cluster_id, dtype=global_param.p_dtype)

        score = pre_score * score

        return score

    def deepcopy(self):
        if self.predecessor is None:
            copied_predecessor = None
        else:
            copied_predecessor = self.predecessor.deepcopy()
        return SklearnClusterCriterion(self.model, self.cluster_id, copied_predecessor, self.clustering_method)

    def get_json(self) -> dict:
        if self.predecessor is None:
            json_predecessor = None
        else:
            json_predecessor = self.predecessor.get_json()

        # This is not an output that may be used to reproduce partitioning.
        return {"type": "sklearn_" + str(self.clustering_method), "predecessor": json_predecessor,
                "cluster_id": self.cluster_id}


class SingleSklearnClusterCriterion(SklearnClusterCriterion):
    def __init__(self, model, cluster_id, predecessor: pm.PartitionCriterion, clustering_method: ClusteringMethod):
        super(SingleSklearnClusterCriterion, self).__init__(model, cluster_id, predecessor, clustering_method)

    def get_score(self, x_vector: tf.Tensor) -> tf.Tensor:
        return tf.cast(1, dtype=global_param.p_dtype)

    def deepcopy(self):
        if self.predecessor is None:
            copied_predecessor = None
        else:
            copied_predecessor = self.predecessor.deepcopy()
        return SklearnClusterCriterion(self.model, self.cluster_id, copied_predecessor, self.clustering_method)

    def get_json(self) -> dict:
        if self.predecessor is None:
            json_predecessor = None
        else:
            json_predecessor = self.predecessor.get_json()

        # This is not an output that may be used to reproduce partitioning.
        return {"type": "sklearn_" + str(self.clustering_method), "predecessor": json_predecessor,
                "cluster_id": self.cluster_id, "is_dummy": True}


class SklearnPartitioningModel(pm.PartitioningModel):
    def __init__(self, ignored_dimensions: List[int], clustering_method: ClusteringMethod):
        super(SklearnPartitioningModel, self).__init__(pm.PartitioningClass.SELF_SUFFICIENT, ignored_dimensions)
        self.max_window_size = 250
        self.clustering_method = clustering_method
        self.model = None

    def automatic_init_criteria(self, data_input: di.DataInput, number_of_partitions: int = None,
                                predecessor_criterion: pm.PartitionCriterion = None):
        clusters_k: int
        if number_of_partitions is not None and number_of_partitions > 1:
            clusters_k = number_of_partitions
        else:
            clusters_k = data_input.n_train // self.max_window_size

        partitioning: List[SklearnClusterCriterion] = []

        if clusters_k >= 2:
            filtered_x_train = self.filter_data_by_ignored_dimensions(data_input.data_x_train)

            logging.info(
                f"Starting determining Partitioning Criteria by Sklearn {str(self.clustering_method)}. "
                f"k={clusters_k}, shape={str(filtered_x_train.shape)}")

            start_time = time.time()

            self.model = self.get_model(clusters_k)
            fit_result = self.model.fit_predict(filtered_x_train)

            logging.info(f"Finished partitioning in {time.time() - start_time}s")

            logging.info("Initializing partition criteria by means of previously determined cluster centers.")
            for i in range(fit_result.min(), fit_result.max() + 1):
                partitioning.append(SklearnClusterCriterion(self.model, i, predecessor_criterion, self.clustering_method))
        else:
            partitioning.append(SingleSklearnClusterCriterion(None, 0, predecessor_criterion, self.clustering_method))

        self.init_partitioning(partitioning)

    def get_model(self, number_of_partitions: int):
        if self.clustering_method is ClusteringMethod.GAUSSIAN_MIXTURE:
            return self.get_model_gaussian_mixture(number_of_partitions)
        elif self.clustering_method is ClusteringMethod.BIRCH:
            return self.get_model_birch(number_of_partitions)
        else:
            raise Exception(f"Invalid SklearnClusteringMethod: {self.clustering_method}")

    def get_model_gaussian_mixture(self, number_of_partitions: int):
        model = Birch(n_clusters=number_of_partitions, threshold=0.1 / number_of_partitions)
        return model

    def get_model_birch(self, number_of_partitions: int):
        model = GaussianMixture(n_components=number_of_partitions)
        return model

    def deepcopy(self):
        partitioning: List[SklearnClusterCriterion] = [pc.deepcopy() for pc in self.partitioning]
        copied_self = SklearnPartitioningModel(self.ignored_dimensions.copy(), self.clustering_method)
        copied_self.init_partitioning(partitioning)
        return copied_self
