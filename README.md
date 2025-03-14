# EM-algorithm
Implement the EM algorithm for fitting a Gaussian mixture model for the MNIST handwritten digits dataset.

We will use only two digits "2" and "6". We fit the GMM model with C=2. There are 1,990 images, and each column of the matrix corresponds to one image. First, we use PCA to reduce the dimensionality of the data before applying EM. We perform K-means clustering with K = 2 and compute the misclassification rates for the digits "2" and "6" separately. Finally, we compare these results with those obtained using GMM.
