# Gaussian Process Model Inference
This repository encompasses code to automatically infer Gaussian Process Models specifically tailored for an 
arbitrarily predefined dataset. A Gaussian Process Model describes a Gaussian Process defined by its specific mean 
and covariance function accompanied by their respective hyperparameters [1]. In earlier publications Gaussian Process 
Model Inference was also called Gaussian Process Model Retrieval.

## Algorithms
We proposed the concept of Gaussian Process Model Inference in detail in [2], while there have already been early 
approaches to solve that issue: CKS [3], ABCD [4], SKC [5]. Although SKC provides a significant improvement on runtime performance,
it still only allows to analyze up to 100k records [5]. Therefore, we developed different solutions based on local 
Gaussian Process approximations to tackle that big data problem:

- Concatenated Composite Covariance Search (3CS) algorithm [6]
    - sequentially traverses input data to build one locally specialized model
    - partitions result from change point detection
    - in [6], we used an earlier version (v1) for evaluation based on Tensorflow 2.0
- Large-Scale Automatic Retrieval of GPMs (LARGe) algorithm [2]
    - iteratively optimizes sub-model per defined data partition
        - partitions result from either clustering, change point detection or other partitioning methods
    - [2] is a short paper proposing the concept of GPM inference and the LARGe algorithm
    - in [2], we used an earlier version (v1) for evaluation based on Tensorflow 2.0
-  Lineage  GPM  Inference (LGI) [7]
    - top-down approach for inferring GPMs, that utilizes both local and global approximations
    - partitions either result from change point detection (univariate data) or clustering (multivariate data)
    - in [7], we used a earlier version (v1) for evaluation based on Tensorflow 2.3

## Implementation
We implemented the given Algorithms and the general problem of automatic GPM inference using Python 3.9, Tensorflow 2.7 
and Tensorflow-Probability 0.14 (and other auxiliary libraries). We encapsulated the basic Gaussian Process Model 
functionalities in a separate project/library called [gpbasics](https://github.com/Bernsai/GaussianProcessFundamentals). 

## References
[1] F. Berns and C. Beecks. 2020. Towards Large-scale Gaussian Process
Models for Efficient Bayesian Machine Learning. In Proceedings of the 9th
International Conference on Data Science, Technology and Applications.

[2] F. Berns and C. Beecks, Automatic gaussian process
model retrieval for big data, in CIKM, ACM, 2020.

[3] D. Duvenaud, J. R. Lloyd, R. B. Grosse, J. B.
Tenenbaum, and Z. Ghahramani, Structure discovery in nonparametric regression through compositional
kernel search, in ICML (3), vol. 28 of JMLR Workshop and Conference Proceedings, JMLR.org, 2013,
pp. 1166-1174.

[4] J. R. Lloyd, D. Duvenaud, R. B. Grosse, J. B. Tenenbaum,
and Z. Ghahramani. 2014. Automatic Construction and Natural-Language
Description of Nonparametric Regression Models. In AAAI. 1242–1250.

[5] H. Kim and Y. W. Teh, “Scaling up the Automatic Statistician: Scalable
structure discovery using Gaussian processes,” in Proceedings of the
21st International Conference on Artificial Intelligence and Statistics,
vol. 84, 2018, pp. 575–584.

[6] F. Berns, K. Schmidt, I. Bracht, and C. Beecks,
3CS Algorithm for Efficient Gaussian Process Model
Retrieval, 25th International Conference on Pattern
Recognition (ICPR), 2020.

[7] F. Berns and C. Beecks, Complexity-Adaptive Gaussian Process Model Inference for Large-Scale Data, 
in SDM, SIAM, 2021.
