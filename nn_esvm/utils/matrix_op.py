import numpy as np
import torch

# taken from https://github.com/svsamsonov/esvm/blob/master/baselines.py
def standartize(X_train, X_test, intercept=True):
    if intercept:#adds intercept term
        X_train = np.concatenate((np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train),axis=1)
        if X_test is not None:
            X_test = np.concatenate((np.ones(X_test.shape[0]).reshape(X_test.shape[0],1),X_test),axis=1)
    #d = X_train.shape[1]
    # Centering the covariates
    means = np.mean(X_train,axis=0)
    if intercept:#do not subtract the mean from the bias term
        means[0] = 0.0
    # Normalizing the covariates
    X_train -= means
    Cov_matr = np.dot(X_train.T,X_train)
    U,S,V_T = np.linalg.svd(Cov_matr,compute_uv = True)
    #Sigma_half = U @ np.diag(np.sqrt(S)) @ V_T
    Sigma_minus_half = U @ np.diag(1./np.sqrt(S)) @ V_T
    X_train = X_train @ Sigma_minus_half
    # The same for test sample
    if X_test is not None:
        X_test = (X_test - means) @ Sigma_minus_half
        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    return torch.tensor(X_train, dtype=torch.float32), None


def poor_standartize(X_train, X_test, intercept=True):
    if intercept:#adds intercept term
        X_train = np.concatenate((np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train),axis=1)
        if X_test is not None:
            X_test = np.concatenate((np.ones(X_test.shape[0]).reshape(X_test.shape[0],1),X_test),axis=1)
    #d = X_train.shape[1]
    # Centering the covariates
    means = np.mean(X_train,axis=0)
    stds = np.std(X_train, axis=0)
    if intercept:#do not subtract the mean from the bias term
        means[0] = 0.0
    # Normalizing the covariates
    X_train -= means
    #Sigma_half = U @ np.diag(np.sqrt(S)) @ V_T
    X_train = X_train/stds[None, :]
    # The same for test sample
    if X_test is not None:
        X_test = (X_test - means) / stds[None, :]
        return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    return torch.tensor(X_train, dtype=torch.float32), None
