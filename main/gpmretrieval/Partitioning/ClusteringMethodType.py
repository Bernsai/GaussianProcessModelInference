from enum import Enum


class ClusteringMethod(Enum):
    GAUSSIAN_MIXTURE = 0
    BIRCH = 1
    KMEANS = 2
