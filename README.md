# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0


# Data-Augmentation-by-Guided-Deep-Interpolation
A novel data augmentation method to improve imbalanced data sets for supervised learning tasks.
The architecture relies on a deep subspace clustering auto-encoder network and an interpolation method in the latent space of the network.
Experiments and validation are performed using CNN architectures.
For the details of the proposed method the authors refer to G. Szlobodnyik, F. Lóránt, "Data Augmentation by Guided Deep Interpolation", submitted to Applied Soft Computing.

Abstarct

State-of-the-art machine learning algorithms require large amount of high quality data. In practice, however, the sample size is commonly low and data is imbalanced along different class labels. Low sample size and imbalanced class distribution can significantly deteriorate the predictive performance of machine learning models. In order to overcome data quality issues, we propose a novel data augmentation method, Guided Deep Interpolation (GDI). It is based on a convolutional auto-encoder network, which is equipped with an auxiliary linear self-expressive layer. The network is trained by minimizing a composite objective function so that to extract the underlying  clustered structure of semantic similarities of data points while high reconstruction quality is also preserved. The trained network is used to define a sampling strategy and a synthetic data generation procedure. Making use of the weights of the self-expressive layer, we introduce a measure of semantic variability to quantify how similar a data point to other data points on average. Based on the proposed measure of semantic variability, a joint distribution is defined. Using the distribution we can draw pairs of similar data points so that one point is semantically underrepresented (isolated) while its pair possesses relatively high semantic variability. A sampled pair is interpolated in the deep feature space of the network so that to increase semantic variability while preserve class label of the semantically underrepresented data point. The trained decoder is used to determine pixel space representations of latent space interpolations. The resulting data augmentation procedure generates synthetic samples by increasing the semantic variability of semantically underrepresented instances in a class label preserving way.


![autoencoder_se_extended](https://user-images.githubusercontent.com/19731435/117464733-3e750400-af51-11eb-9608-e2e46cc2d92e.jpg)
