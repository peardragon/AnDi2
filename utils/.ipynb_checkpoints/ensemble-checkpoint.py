import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

### single run label + traj reorg
def separate_label_values(pred):
    '''
    Given a prediction over trjaectory segments, extracts the predictions for each segment property
    as well as the changepoint values.
    '''        
    Ds = pred[1::4]
    alphas = pred[2::4]
    states = pred[3::4]
    cp = pred[4::4]    
    return Ds, alphas, states, cp

def read_and_process_data(exp_dir):
    model_dict = {'single_state':0, 'multi_state':1, 'immobile_traps':2, 'dimerization':3, 'confinement':4}

    a_list = []
    k_list = []
    s_list = []
    cp_list = []

    file_list = os.listdir(exp_dir)
    file_list_fov = [file for file in file_list if file.startswith("fov")]
    for file in tqdm(file_list_fov):
        #################################################
        with open(exp_dir + file, "r") as f:
            label_lines = f.read().splitlines()
        
        
        columns = ['traj_idx', 'model', 'Ds', 'alphas', 'states', 'changepoints']                
        for line in label_lines:

                
            # Extract values with comma separator and transform to float
            label_traj = line.split(',')
            label = [float(i) for i in label_traj]
            
            k, a, s, cp = separate_label_values(label)
            prev = 0
            for index in range(len(cp)):
                const = int(cp[index]-prev)
                k_list.extend([k[index]]*const)
                a_list.extend([a[index]]*const)
                s_list.extend([s[index]]*const)
                
                cp_list.append(cp)
                prev = cp[index]

    return a_list, k_list, s_list, cp_list


class FixedWeightGMM:
    def __init__(self, n_components=1, fixed_weights=None, tol=1e-3, max_iter=100, random_state=None):
        self.n_components = n_components
        self.fixed_weights = fixed_weights if fixed_weights is not None else np.ones(n_components) / n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.means_ = None
        self.variances_ = None
        self.weights_ = self.fixed_weights

    def _initialize_parameters(self, X):
        n_samples, _ = X.shape
        self.means_ = X[self.random_state.choice(n_samples, self.n_components, replace=False)]
        self.variances_ = np.var(X, axis=0) * np.ones(self.n_components)

    def _e_step(self, X):
        log_resp = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = self._estimate_log_gaussian_prob(X, self.means_[k], self.variances_[k])
        log_resp += np.log(self.weights_)
        log_prob_norm = np.log(np.sum(np.exp(log_resp), axis=1))
        log_resp -= log_prob_norm[:, np.newaxis]
        return np.exp(log_resp), np.sum(log_prob_norm)

    def _m_step(self, X, resp):
        nk = resp.sum(axis=0)
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.variances_[k] = np.dot(resp[:, k], (diff ** 2).sum(axis=1)) / nk[k]

    def _estimate_log_gaussian_prob(self, X, mean, variance):
        variance = np.maximum(variance, 1e-6)  # avoid division by zero
        return -0.5 * ((X - mean) ** 2 / variance + np.log(2 * np.pi * variance)).sum(axis=1)

    def fit(self, X):
        self._initialize_parameters(X)
        lower_bound = -np.infty
        self.converged_ = False

        for n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound
            resp, log_prob_norm = self._e_step(X)
            lower_bound = log_prob_norm
            self._m_step(X, resp)

            if abs(lower_bound - prev_lower_bound) < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = n_iter
        self.lower_bound_ = lower_bound
        return self

    def get_params(self):
        return self.means_, self.variances_, self.weights_


def _estimate_log_gaussian_prob(X, mean, variance):
    return -0.5 * ((X - mean) ** 2 / variance + np.log(2 * np.pi * variance)).sum(axis=1)

def score_samples(X, n_components, means, variances, weights):
    log_likelihood = np.zeros(X.shape[0])
    for k in range(n_components):
        log_likelihood += weights[k] * np.exp(_estimate_log_gaussian_prob(X, means[k], variances[k]))
    return np.log(log_likelihood)
    
# Predict the cluster for each sample
def plot_gmm(data, n_state, means, covs, weights):
    data = np.array(data).reshape(-1, 1)
    # Plot the results
    x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
    

    logprob = score_samples(x, n_state, means, covs, weights)
    pdf = np.exp(logprob)
    
    plt.hist(data, bins=30, density=True, alpha=0.5, color='gray')
    plt.plot(x, pdf, '-k', label='GMM fit')
    
    for mean, covariance in zip(means, covs):
        plt.plot(x, 1/np.sqrt(2*np.pi*covariance) * np.exp(-(x - mean)**2 / (2*covariance)), label=f'Component mean: {mean:.2f}')
    
    plt.title('Gaussian Mixture Model')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def get_ensemble_results(exp_dir, max_iter=2500, threshold = 0.01, verbose=False):
    a, k, s, cp = read_and_process_data(exp_dir)
    n_state = np.sum(np.unique(s, return_counts=True)[-1] > len(s)*threshold) #threshold
    

    results = np.zeros((5, n_state))
    weights = np.zeros((2, n_state))
    metric = []
    weights_init = []
    bics = []
    for index, data in enumerate([a,k]):
        data = np.array(data).reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_state, random_state=42,
                              max_iter=max_iter, verbose=False, tol=1e-6)
        gmm.fit(data)
        bics.append(gmm.bic(data))
        weights_init.append(gmm.weights_)
    
        mean, cov, weight = gmm.means_, gmm.covariances_, gmm.weights_
    
        results[0+index*2, :] = mean.flatten()
        results[1+index*2, :] = cov.flatten()
        weights[index, :] = weight.flatten()
    
    weights_init = weights_init[np.array(bics).argsort()[0]]
    index = np.array(bics).argsort()[1]
    
    data = np.array([a,k])[index].reshape(-1, 1)
    fixed_gmm = FixedWeightGMM(n_components=n_state, fixed_weights=weights_init,
                               random_state=42,
                               tol=1e-6, max_iter=max_iter)
    
    fixed_gmm.fit(data)
    mean, cov, weight = fixed_gmm.get_params()
    
    weights[index, :] = weight.flatten()
    results[0+index*2, :] = mean.flatten()
    results[1+index*2, :] = cov.flatten()
    weights_init = weight
    
    results[-1,:] = np.mean(weights, axis=0)

    if verbose:
        for index, data in enumerate([a,k]):
            data = np.array(data).reshape(-1, 1)
            mean = results[0+index*2, :]
            cov = results[1+index*2, :]
            weight = results[-1,:]
            
            plot_gmm(data, n_state, mean, cov, weight)
    return results
