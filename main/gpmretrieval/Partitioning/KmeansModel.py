import gpbasics.global_parameters as global_param

global_param.ensure_init()

from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import List
import gpbasics.KernelBasics.PartitioningModel as pm
import gpbasics.DataHandling.DataInput as di
import time
import logging
import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def cluster_center_score(x_vector, cluster_center):
    return tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x_vector, cluster_center), axis=1))


class ClusterCenterCriterion(pm.PartitionCriterion):
    def __init__(self, cluster_center: tf.Tensor):
        assert len(cluster_center.shape) == 1, "Invalid cluster center given: shape=%s. " \
                                               "Shape of cluster center has to be [d,]." % str(cluster_center.shape)
        super(ClusterCenterCriterion, self).__init__(pm.PartitioningClass.SMALLEST_DISTANCE)
        self.cluster_center: tf.Tensor = cluster_center

    def get_score(self, x_vector: tf.Tensor) -> tf.Tensor:
        return cluster_center_score(x_vector, self.cluster_center)

    def deepcopy(self):
        return ClusterCenterCriterion(self.cluster_center.copy())

    def get_json(self) -> dict:
        return {"type": "cluster_center", "center": self.cluster_center.numpy().tolist()}

    def __hash__(self):
        return hash(tuple(self.cluster_center.tolist()))


class KMeansModel(pm.PartitioningModel):
    def __init__(self, ignored_dimensions: List[int]):
        super(KMeansModel, self).__init__(pm.PartitioningClass.SMALLEST_DISTANCE, ignored_dimensions)
        self.max_window_size = 250
        self.model = None

    def automatic_init_criteria(self, data_input: di.DataInput, number_of_partitions: int = None,
                                predecessor_criterion: pm.PartitionCriterion = None):
        logging.info("Setting up Automatically determining Partitioning Criteria by KMeans")

        clusters_k: int
        if number_of_partitions is not None and number_of_partitions > 1:
            clusters_k = number_of_partitions
        else:
            clusters_k = data_input.n_train // self.max_window_size

        partitioning: List[ClusterCenterCriterion] = []

        if clusters_k >= 2:
            filtered_x_train = self.filter_data_by_ignored_dimensions(data_input.data_x_train)

            logging.info("Starting determining Partitioning Criteria by KMeans. k=%i, shape=%s"
                         % (clusters_k, str(filtered_x_train.shape)))

            start_time = time.time()
            if data_input.n_train < 100000:
                logging.info("Using usual KMeans as dataset is smaller than 100k records.")
                self.model = KMeans(n_clusters=clusters_k, random_state=0, tol=1e-3)
                kmeans = self.model.fit(filtered_x_train)
            else:
                logging.info("Using MiniBatchKMeans as dataset is larger than 100k records.")
                self.model = MiniBatchKMeans(n_clusters=clusters_k, batch_size=max([clusters_k, 100]))
                kmeans = self.model.fit(filtered_x_train)

            logging.info("Finished partitioning in %f s" % (time.time() - start_time))

            logging.info("Initializing partition criteria by means of previously determined cluster centers.")
            for cluster_center in kmeans.cluster_centers_:
                partitioning.append(ClusterCenterCriterion(tf.constant(cluster_center, dtype=global_param.p_dtype)))
        else:
            partitioning.append(ClusterCenterCriterion(tf.reduce_mean(
                tf.constant(data_input.get_x_range(), dtype=global_param.p_dtype), axis=1)))

        self.init_partitioning(partitioning)

    def get_data_record_indices_per_partition(self, x_vector: tf.Tensor) -> List[tf.Tensor]:
        """
        This is an auxiliary method, that translates the scores per PartitionCriterion for the given input data
        (i.e. x_vector) into the corresponding index sets per partition to retrieve the right data per partition
        correspondingly.

        Args:
            x_vector:

        Returns: List of List of indices (where the inner list is represented as tf.Tensor)

        """
        if self.model is None:
            score_matrix_train: List[tf.Tensor] = []

            for criterion in self.partitioning:
                score_matrix_train.append(
                    tf.cast(tf.reshape(criterion.get_score(x_vector), [-1, 1]), tf.float32))

            score_matrix_train: tf.Tensor = tf.concat(score_matrix_train, axis=1)

            if self.partition_class == pm.PartitioningClass.SMALLEST_DISTANCE:
                # This is a workaround for the error of one data record being assigned to multiple partitions
                score_matrix_train = \
                    score_matrix_train + tf.random.normal(
                        score_matrix_train.shape, 0.0, 1e-10, dtype=tf.float32, seed=123)

                col_min = tf.reduce_min(score_matrix_train, axis=1)
                score_matrix_train = score_matrix_train == tf.reshape(col_min, [-1, 1])

            score_matrix_train = tf.cast(score_matrix_train, tf.int8)

            indices_per_partition: List[tf.Tensor] = []

            for i in range(self.get_number_of_partitions()):
                indices = tf.reshape(tf.where(score_matrix_train[:, i] == 1), [-1, ])
                indices_per_partition.append(indices)

            if len(indices_per_partition) == 0:
                indices_per_partition = [tf.cast(tf.linspace(0, len(x_vector) - 1, len(x_vector)), tf.int64)]
        else:
            k = len(self.partitioning)

            cluster_pred = self.model.predict(x_vector)

            indices_per_partition: List[tf.Tensor] = []
            for i in range(k):
                indices = tf.reshape(tf.where(cluster_pred == i), [-1, ])
                indices_per_partition.append(indices)

            cluster_lengths = [len(entry) for entry in indices_per_partition]

            logging.info(f"Cluster found. sizes: min {tf.reduce_min(cluster_lengths)} "
                         f"mean {tf.reduce_mean(cluster_lengths)} max {tf.reduce_max(cluster_lengths)}")

        return indices_per_partition

    def deepcopy(self):
        partitioning: List[ClusterCenterCriterion] = [pc.deepcopy() for pc in self.partitioning]
        copied_self = KMeansModel(self.ignored_dimensions.copy())
        copied_self.init_partitioning(partitioning)
        return copied_self
