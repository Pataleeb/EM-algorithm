import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.io as sio
import seaborn as sns
from scipy.stats import mode
from sklearn.cluster import KMeans


data = sio.loadmat('data.mat')
print(data.keys())
X=data['data']

x=data['data'].T
m,n=x.shape
mean=np.mean(x, axis=0)
std=np.std(x, axis=0)
std[std==0]=1
x_scaled=(x-mean)/std
C=np.matmul(x_scaled.T,x_scaled)/m
C=(C+C.T)/2
d=4
values,V=np.linalg.eigh(C)
ind = np.argsort(values)[::-1][:d]
V = V[:, ind]
pdata=np.dot(x_scaled,V)

##EM algorithm
K=2
seed=12345
np.random.seed(seed)
pi = np.random.random(K)
pi = pi/np.sum(pi)

mu = np.random.randn(K,d)
mu_old = mu.copy()

sigma = []
for i in range(K):

    dummy = np.random.randn(d,d)
    sigma.append(dummy @ dummy.T+1e-6*np.eye(d))

# initialize the posterior
tau = np.full((m, K), fill_value=0.0)
maxIter = 100
tol = 1e-3
log_likelihoods=[]
plt.ion()

pi_values=[]
for ii in range(maxIter):

    # E-step
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
        tau[:, kk] = np.clip(tau[:, kk], 1e-8, None)

    sum_tau = np.sum(tau, axis=1,keepdims=True)
    #sum_tau = np.clip(sum_tau, 1e-8, None)

    tau/=sum_tau

    log_likelihood = np.sum(np.log(sum_tau))
    log_likelihoods.append(log_likelihood)

    # M-step
    for kk in range(K):
        nk=max(np.sum(tau[:, kk]),1e-8)

        pi[kk] = nk / m

        mu[kk] = np.sum(tau[:, kk,np.newaxis]*pdata,axis=0) / nk

        dummy = pdata - mu[kk]
        sigma[kk] = (dummy.T @ (tau[:, kk][:, np.newaxis] * dummy)) / nk

    pi_values.append(pi.copy())

    plt.scatter(pdata[:, 0], pdata[:, 1], c=tau[:, 0], cmap='viridis')
    plt.title(f'Iteration {ii}')
    plt.pause(0.1)


    if ii > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        print("Converged at iteration", ii)
        break

plt.figure()
plt.plot(log_likelihoods, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.title("EM Algorithm Convergence")
plt.show()

###Q2

pi_avaerage=np.mean(pi_values,axis=0)
print("\nAverage weights for each component:")
for k in range(K):
    print(f"Component {k + 1}: Average π = {pi_avaerage[k]:.4f}")

mean_original=mu@V.T
mean_original=mean_original*std+mean

plt.figure(figsize=(8,4))

for k in range(K):
    plt.subplot(1, K, k+1)

    plt.imshow(mean_original[k].reshape(28, 28), cmap='gray')
    plt.title(f"Component {k + 1} Average Image")
    plt.axis('off')
plt.show()

plt.figure(figsize=(8,8))
for k in range(K):
    plt.subplot(1, K, k+1)

    sns.heatmap(sigma[k], annot=True, cmap='gray', fmt=".2f")
    plt.title(f"Covariance Matrix for Component {k + 1}")
plt.show()

###Q3
###Misclassification rate

true_labels=sio.loadmat('label.mat')
print(true_labels.keys())
true_labels=true_labels['trueLabel'].flatten()


gmm_predicted_labels = np.argmax(tau, axis=1)
cluster_0_label=mode(true_labels[gmm_predicted_labels == 0]).mode.item()
cluster_1_label=mode(true_labels[gmm_predicted_labels==1]).mode.item()
mapped_labels_gmm = np.where(gmm_predicted_labels == 0, cluster_0_label, cluster_1_label)

digit2=true_labels==2
digit6=true_labels==6

misclassified_d2=np.mean(mapped_labels_gmm[digit2]!=true_labels[digit2])
misclassified_d6=np.mean(mapped_labels_gmm[digit6]!=true_labels[digit6])

print(f"GMM Misclassification Rate for Digit 2: {misclassified_d2:.4f}")
print(f"GMM Misclassification Rate for Digit 6: {misclassified_d6:.4f}")

##K-means clustering
kmeans=KMeans(n_clusters=2, random_state=seed, n_init='auto')
kmenas_labels=kmeans.fit_predict(pdata)

cluster_0_kmeans=mode(true_labels[kmenas_labels == 0]).mode.item()
cluster_1_kmeans=mode(true_labels[kmenas_labels==1]).mode.item()
mapped_kmeans = np.where(kmenas_labels == 0, cluster_0_kmeans, cluster_1_kmeans)
misclassified_kmeans2 = np.mean(mapped_kmeans[digit2]!=true_labels[digit2])
misclassified_kmeans6=np.mean(mapped_kmeans[digit6]!=true_labels[digit6])

print(f"K-Means Misclassification Rate for Digit 2: {misclassified_kmeans2:.4f}")
print(f"K-Means Misclassification Rate for Digit 6: {misclassified_kmeans6:.4f}")