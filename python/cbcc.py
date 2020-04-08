'''
@author: Edwin Simpson
'''
import sys, logging
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import psi, gammaln, digamma
from ibccdata import DataHandler
from scipy.optimize import fmin, fmin_cobyla
from scipy.stats import gamma, beta as beta_dist, bernoulli
import antoniak

import ibcc 
from sklearn.mixture import BayesianGaussianMixture

class CBCC(ibcc.IBCC):
# Clustering-based IBCC using a Dirichlet process. Each cluster has a single, shared confusion matrix that all workers
# in that cluster use.  
#
# Correspondence with the Moreno notation:
# \Psi --> Pi. 
# \pi --> responsibilities.
# z --> t
# y --> C
# \tau --> \kappa
# Hyperparameters:
# \eta and \beta --> \alpha
# \mu and \epsilon --> \nu
# \alpha --> conc
# a_\alpha --> we don't use these here, just a point value for the concentration?
# b_\alpha --> "
      
# Model parameters and hyper-parameters -----------------------------------------------------------------------------
    def __init__(self, nclasses=2, nscores=2, alpha0=None, nu0=None, conc_prior=1, nclusters=100, 
                 K=1, uselowerbound=False, dh=None):
        super(CBCC, self).__init__(nclasses, nscores, alpha0, nu0, K, uselowerbound, dh)
        
        self.conc_prior = conc_prior # concentration hyperparameter
        self.nclusters = nclusters         

# Initialisation ---------------------------------------------------------------------------------------------------

    def init_weights(self):
        # The weights are the posterior probabilities of each cluster, i.e. proportion of data covered by each cluster
        self.expec_weights()
    
    def init_responsibilities(self):
        # The responsibilities are the posterior probabilities that a worker belongs to a particular cluster 
        self.r = 1.0 / self.nclusters + np.zeros((self.K, self.nclusters))
        self.logr = np.log(self.r)        
        
    def _init_t(self):
        self.init_responsibilities()
        self.init_weights()
        super(CBCC, self)._init_t()
        
    def _init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        '''
        self.alpha0 = self.alpha0.astype(float)
        # if we specify different alpha0 for some agents, we need to do so for all K agents. The last agent passed in 
        # will be duplicated for any missing agents.
        if np.any(self.clusteridxs_alpha0): # map from diags list of cluster IDs
            if not np.any(self.alpha0_cluster):
                self.alpha0_cluster = self.alpha0
                self.alpha0_length = self.alpha0_cluster.shape[2]
            self.alpha0 = self.alpha0_cluster[:, :, self.clusteridxs_alpha0]        
        else:
            if len(self.alpha0.shape)==2:
                self.alpha0  = self.alpha0[:,:,np.newaxis]
                            
            if self.alpha0.shape[2] < self.nclusters:
                # We have a new dataset with more agents than before -- create more priors.
                nnew = self.nclusters - self.alpha0.shape[2]
                alpha0new = self.alpha0[:, :, 0]
                alpha0new = alpha0new[:, :, np.newaxis]
                alpha0new = np.repeat(alpha0new, nnew, axis=2)
                self.alpha0 = np.concatenate((self.alpha0, alpha0new), axis=2)
                
        # Make sure self.alpha is the right size as well. Values of self.alpha not important as we recalculate below
        self.alpha0 = self.alpha0[:, :, :self.nclusters] # make this the right size if there are fewer classifiers than expected
        self.alpha = np.zeros((self.nclasses, self.nscores, self.nclusters), dtype=np.float) + self.alpha0
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K)) 
        self._expec_lnPi(posterior=False) # calculate alpha from the initial/prior values only in the first iteration

# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------

    def expec_weights(self):
        #nk is the sum of responsibilities for the agents, shape (nclusters,)
        #nk[::-1] reverses the order
        #np.cumsum(nk[::-1]) gives the total weights for clusters with ID >= i for each index i in the array
        # (np.cumsum(nk[::-1])[-2::-1], 0) flips the order back and skips the first value so that we have the total 
        #weight for clusterIDs > i
        #np.hstack((blah..., 0)) appends 0 to the end since zero weight for cluster IDs > the last cluster
        # resulting weight_concentration_ has two values, effectively parameters to a beta distribution where [0] is 
        # the probability of this cluster and 
        
        nk = np.sum(self.r, axis=0)
        
        weight_concentration_ = (1. + nk, (self.conc_prior + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
        self.weight_conc = np.array(weight_concentration_)
        # estimate_log_weights
        
        digamma_sum = digamma(weight_concentration_[0] + weight_concentration_[1])
        digamma_a = digamma(weight_concentration_[0])
        
        logp_current_vs_subsequent_clusters = digamma_a - digamma_sum
        
        digamma_b = digamma(weight_concentration_[1])
        logp_subsequent_vs_current = digamma_b - digamma_sum
        logp_subsequent = np.cumsum(logp_subsequent_vs_current)[:-1]
        logp_current_or_subsequent = np.hstack((0, logp_subsequent))  
        
        self.Elogp_clusters = np.array((logp_current_vs_subsequent_clusters, logp_subsequent_vs_current))
        self.logw = logp_current_vs_subsequent_clusters + logp_current_or_subsequent

    def _expec_t(self):
        super(CBCC, self)._expec_t()

    def expec_responsibilities(self):
        # for each cluster value, compute the log likelihoods of the data. This will be a sum  
        # over lnPi columns for the observed labels multiplied by E_t and summed over all classes and all data points
        # to get logp(C^{(k)} | cluster_k = cluster)        
        loglikelihoods = np.zeros((self.K, self.nclusters))
        
        self.cluster_lnPi = np.zeros((self.nclasses, self.nscores, self.nclusters))         
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for s in range(self.nscores):
            self.cluster_lnPi[:, s, :] = digamma(self.alpha[:, s, :]) - psiSumAlpha
        
        for cl in range(self.nclusters):
            for j in range(self.nclasses):
                data = []
                for l in range(self.nscores):
                    if self.table_format_flag:
                        data_l = self.C[l] * self.cluster_lnPi[j, l, cl]
                    else:   
                        data_l = self.C[l].multiply(self.cluster_lnPi[j, l, cl])                        
                    data = data_l if data==[] else data+data_l
                loglikelihoods[:, cl:cl+1] += data.T.dot(self.E_t[:, j][:, np.newaxis])
        
        logweights = self.logw[np.newaxis, :]
        
        weighted_log_prob = loglikelihoods + logweights
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        self.lnr = log_resp
        self.r = np.exp(self.lnr)

    def _train_alpha_counts(self):
        # Save the counts from the training data so we only recalculate the test data on every iteration
        if not len(self.alpha_tr):
            self.alpha_tr = np.zeros(self.alpha.shape)
            if self.Ntrain:
                for j in range(self.nclasses):
                    for l in range(self.nscores):
                        for cl in range(self.nclusters):
                            Tj = self.E_t[self.trainidxs, j].reshape((self.Ntrain, 1))
                            if self.table_format_flag:
                                self.alpha_tr[j,l,cl] = np.sum(                                
                                   (self.C[l][self.trainidxs,:] * self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1) )
                            else:
                                self.alpha_tr[j,l,cl] = np.sum( 
                                   self.C[l][self.trainidxs,:].multiply(self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1) )
                            
            self.alpha_tr += self.alpha0
            
    def _post_Alpha(self):  # Posterior Hyperparams
        self._train_alpha_counts()
        
        # Add the counts from the test data
        for j in range(self.nclasses):
            for l in range(self.nscores):
                Tj = self.E_t[self.testidxs, j].reshape((self.Ntest, 1))
                for cl in range(self.nclusters):
                    if self.table_format_flag:
                        counts = (self.Ctest[l] * self.r[:, cl][np.newaxis, :]).T.dot(Tj).reshape(-1)
                    else:
                        counts = (self.Ctest[l].multiply(self.r[:, cl][np.newaxis, :])).T.dot(Tj).reshape(-1)
                        
                    self.alpha[j, l, cl] = self.alpha_tr[j, l, cl] + np.sum(counts)

    def _expec_lnPi(self, posterior=True):
        self.expec_responsibilities()
        self.expec_weights()
        
        # check if E_t has been initialised. Only update alpha if it has. Otherwise E[lnPi] is given by the prior
        if np.any(self.E_t) and posterior:
            self._post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for j in range(self.nclasses):
            for s in range(self.nscores): 
                self.lnPi[:, s, :] = (psi(self.alpha[:, s, :]) - psiSumAlpha)[np.newaxis, :].dot(self.r.T)
 
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def _post_lnpi(self):
        x = np.sum((self.alpha0-1) * self.cluster_lnPi,1)
        z = gammaln(np.sum(self.alpha0,1)) - np.sum(gammaln(self.alpha0),1)
        
        #cluster weights
        weight_prior_params = np.array([1.0, self.conc_prior])[:, np.newaxis]
        w_x = np.sum((weight_prior_params - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(weight_prior_params, axis=0)) - np.sum(gammaln(weight_prior_params), axis=0))  
        
        # responsibilities
        logp_membership = np.sum(self.r * self.logw[np.newaxis, :])
        
        return np.sum(x+z) + w_x + w_z + logp_membership
                    
    def _q_lnPi(self):
        x = np.sum((self.alpha-1) * self.cluster_lnPi,1)
        z = gammaln(np.sum(self.alpha,1)) - np.sum(gammaln(self.alpha),1)

        #cluster weights        
        w_x = np.sum((self.weight_conc - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(self.weight_conc, axis=0)) - np.sum(gammaln(self.weight_conc), axis=0))  
        
        #responsibilities
        logq_membership = np.sum(self.r * self.lnr)
        
        return np.sum(x+z) + w_x + w_z + logq_membership
    
class HCBCC(CBCC):
# Uses a hierarchical prior over the confusion matrices for each cluster, meaning that all workers in the same 
# cluster share a confusion matrix prior, but their actual confusion matrices can vary.
#
# Correspondence with the Moreno notation:
# \Psi --> Pi. 
# \pi --> responsibilities.
# z --> t
# y --> C
# \tau --> \kappa
# Hyperparameters:
# \eta and \beta --> same, but we still transform the posterior values to \alpha
# \gamma and \phi --> same; Dirichlet parameters for the prior over \eta, the confusion matrix mean; constant for all clusters
# a and b --> same; gamma parameters for the prior over \beta (conf. matrix precision); constant for all clusters
# \mu and \epsilon --> \nu
# \alpha --> conc
# a_\alpha --> we don't use these here, just a point value for the concentration?
# b_\alpha --> "

    def __init__(self, nclasses=2, nscores=2, phi0=None, gamma0=None, nu0=None, cluster_prec_shape=2, 
                 cluster_prec_scale=1, conc_prior=1, nclusters=100, nworkers=1, uselowerbound=False, dh=None):
        super(HCBCC, self).__init__(nclasses=nclasses, nscores=nscores, alpha0=np.array([]), nu0=nu0, conc_prior=1, 
                                    nclusters=100, K=nworkers, uselowerbound=uselowerbound, dh=dh)
        if dh != None:
            self.phi0 = dh.phi0.astype(float)
            self.gamma0 = dh.gamma0.astype(float)
            self.a0 = float(dh.a0)
            self.b0 = float(dh.b0)
        else:
            self.phi0 = np.array(phi0).astype(float)
            self.gamma0 = gamma0.astype(float)
            self.a0 = np.array(cluster_prec_shape).astype(float)
            self.b0 = np.array(cluster_prec_scale).astype(float)
            
    def init_responsibilities(self):
        self.r = 1.0 / self.nclusters + np.zeros((self.K, self.nclusters))
        self.logr = np.log(self.r)     
        
    def _init_lnPi(self):
        '''
        Always creates new self.alpha and self.lnPi objects and calculates self.alpha and self.lnPi values according to 
        either the prior, or where available, values of self.E_t from previous runs.
        
        Here we have several alpha objects:
           - : the alphas for the clusters
           - : the prior hyperhyperparameters for the cluster alphas
           - alpha: the posterior alphas for each worker
        '''
        # Now translate alpha0 into mean and precision for the prior over alpha
        self.gamma0 = self.gamma0[:, :, np.newaxis]
        self.phi0 = self.phi0[:, np.newaxis, np.newaxis]
            
        # Now we can initialise eta
        self.eta = np.repeat((self.gamma0 * self.phi0) / np.sum(self.gamma0 * self.phi0, axis=1)[:, np.newaxis, :], 
                             self.nclusters, axis=2)
        self.phigamma = np.zeros((self.nclasses, self.nscores, self.nclusters))
        # Initialise beta, the precision across the cluster
        self.beta = np.zeros((self.nclasses, 1, self.nclusters), dtype=np.float) + (self.a0[:, np.newaxis, np.newaxis] 
                                                                                / self.b0[:, np.newaxis, np.newaxis])
        
        # The prior per-cluster parameters are eta and beta, and each individual worker is drawn using those priors.
        # The workers have individual posteriors with variational parameters alpha. Initialise by assuming workers have
        #equal probability of each cluster:
        self.alpha_tr = np.zeros((self.nclasses, self.nscores, self.K)) + np.sum(self.eta * self.beta / self.nclusters, 
                                                                                 axis=2)[:, :, np.newaxis]
        self.alpha = self.alpha_tr.copy()
        self.lnPi = np.zeros((self.nclasses, self.nscores, self.K)) + \
                        np.log(self.alpha_tr / np.sum(self.alpha_tr, axis=1)[:, np.newaxis, :])
        self._expec_lnPi(posterior=False) # calculate alpha from the initial/prior values only in the first iteration

# Posterior Updates to Hyperparameters --------------------------------------------------------------------------------
    def expec_responsibilities(self):
        # Use the dirichlet likelihood for cluster rather than the categorical likelihood.         
        # For each cluster value, compute the log likelihoods of the data. 
        # This will be a sum of log Dirichlet PDFs given by _q_lnPi   
        # summed over all classes
        # to get logp( pi^k | cluster_k = cluster)        
        loglikelihoods = np.zeros((self.K, self.nclusters))
        
        for cl in range(self.eta.shape[2]):
            etabeta = (self.eta[:, :, cl] * self.beta[:, :, cl])[:, :, np.newaxis]
            x = np.sum((etabeta-1) * self.lnPi, 1)
            z = gammaln(np.sum(etabeta, 1)) - np.sum(gammaln(etabeta),1)
            loglikelihoods[:, cl] = np.sum(x + z, axis=0)
        
        logweights = self.logw[np.newaxis, :]
        
        weighted_log_prob = loglikelihoods + logweights
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        self.lnr = log_resp
        self.r = np.exp(self.lnr)

    def _train_alpha_counts(self):
        prior_pseudocounts = np.zeros(self.alpha.shape)
        for j in range(self.nclasses):
            prior_pseudocounts[j, :, :] = (self.eta[j, :, :] * self.beta[j, :, :]).dot(self.r.T)
        ibcc.IBCC._train_alpha_counts(self, prior_pseudocounts)

    def _post_Alpha(self):  # Posterior Hyperparams
        self.alpha_tr = [] # reset this so we update with new eta and beta
        ibcc.IBCC._post_Alpha(self)
        
    def _expec_lnPi(self, posterior=True):
        self.expec_responsibilities()
        self.expec_weights()
        
        # check if E_t has been initialised. Only update alpha if it has. Otherwise E[lnPi] is given by the prior
        if np.any(self.E_t) and posterior:
            self._post_Alpha()
        sumAlpha = np.sum(self.alpha, 1)
        psiSumAlpha = psi(sumAlpha)
        for j in range(self.nclasses):
            for s in range(self.nscores): 
                self.lnPi[:, s, :] = (psi(self.alpha[:, s, :]) - psiSumAlpha)[np.newaxis, :] #.dot(self.r)
        
        # need to update the cluster pseudo-count distributions first to get new expected eta and beta
        #translate \eta and \beta to \alpha.
        worker_counts =  self.alpha - self.alpha_tr #the counts for each worker
        self.a = np.zeros((self.nclasses, self.nscores, self.nclusters))
        self.b = np.zeros((self.nclasses, self.nscores, self.nclusters))
        
        for j in range(self.nclasses):
            # v_j^(k) ~ Beta( \beta_j^{q_k} ), where q_k is the cluster ID of worker k
            logv_j = psi(self.beta[j, :, :].dot(self.r.T)) - psi(self.beta[j, :, :].dot(self.r.T) + np.sum(worker_counts[j, :, :], axis=0)[np.newaxis, :])
        
            # s^(k)_{j, l} ~ Antoniak( n^(k)_{j, l}, \beta_j^{q_k} \eta_{j, l}^{q_k} )
            #The exact computation of the expected number of tables is given in: 
            # A Note on the Implementation of Hierarchical Dirichlet Processes, Phil Blunsom et al.
            #The antoniak distribution is explained in: Distributed Algorithms for Topic Models, David Newman et al.
            s_j = np.zeros((self.nscores, self.nclusters))
            for l in range(self.nscores):            
                counts = worker_counts[j, l, :][:, np.newaxis]
                conc = (self.beta[j, 0, :] * self.eta[j, l, :])[np.newaxis, :]
                # For the updates to eta and beta, we take an expectation of ln p(s^(k)_{j, l}) over cluster membership of k by 
                # computing s^(k) using a weighted sum with weights p(q_k = m)
                # -- this follows from the equations in Moreno and Teh
                s_jl = conc * (psi(conc + counts) - psi(conc)) # nclusters x K
                s_j[l, :] = np.sum(s_jl * self.r, axis=0)
                
            # \eta_j^(m) ~ Dir( sum_{ k where q_k=m } s^(k)_{j, .} + \phi_j \gamma_j )
            # We need to determine expectation of \eta
            self.phigamma[j, :, :] = s_j + self.phi0[j, :, :] * self.gamma0[j, :, :]
            self.eta[j, :, :] = self.phigamma[j, :, :] / np.sum(self.phigamma[j, :, :], axis=0)[np.newaxis, :] 
         
            # \beta_j^(k) ~ Gamma( sum_{k where q_k=m} sum_{l} s_{j, l}^(k) + a_j, b_j - sum_{k where q_k=m} log(v_{j}^(k) ) )
            # we need expectation of beta
            self.a[j, :, :] = np.sum(s_j, axis=0) + self.a0[j]
            self.b[j, :, :] = self.b0[j] - logv_j.dot(self.r)
        self.beta = self.a / self.b 
        
# Likelihoods of observations and current estimates of parameters --------------------------------------------------
    def _post_lnpi(self):
        x_eta = np.sum((self.phi0*self.gamma0 - 1) * self.eta, 1)
        z_eta = (gammaln(np.sum(self.phi0*self.gamma0, 1)) - np.sum(gammaln(self.phi0*self.gamma0), 1))[:, np.newaxis]
        
        lnp_beta = gamma.logpdf(self.beta, self.a0[:, np.newaxis, np.newaxis], scale=self.b0[:, np.newaxis, np.newaxis])
        
        cluster_params = (self.eta * self.beta).dot(self.r.T)
        x = np.sum((cluster_params - 1) * self.lnPi, 1)
        z = gammaln(np.sum(cluster_params, 1)) - np.sum(gammaln(cluster_params), 1)
        
        #cluster weights
        weight_prior_params = np.array([1.0, self.conc_prior])[:, np.newaxis]
        w_x = np.sum((weight_prior_params - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(weight_prior_params, axis=0)) - np.sum(gammaln(weight_prior_params), axis=0))  
        
        # responsibilities
        logp_membership = np.sum(self.r * self.logw[np.newaxis, :])
        
        return np.sum(x_eta + z_eta) + np.sum(lnp_beta) + w_x + w_z + logp_membership + np.sum(x + z)
                    
    def _q_lnPi(self):
        x_eta = np.sum((self.phigamma - 1) * self.eta, 1)
        z_eta = (gammaln(np.sum(self.phigamma, 1)) - np.sum(gammaln(self.phigamma), 1))[:, np.newaxis]
        
        lnq_beta = gamma.logpdf(self.beta, self.a, scale=self.b)
        
        x = np.sum((self.alpha - 1) * self.lnPi, 1)
        z = gammaln(np.sum(self.alpha, 1)) - np.sum(gammaln(self.alpha), 1)

        #cluster weights        
        w_x = np.sum((self.weight_conc - 1) * self.Elogp_clusters)
        w_z = np.sum(gammaln(np.sum(self.weight_conc, axis=0)) - np.sum(gammaln(self.weight_conc), axis=0))  
        
        #responsibilities
        logq_membership = np.sum(self.r * self.lnr)
        
        return np.sum(x_eta + z_eta) + np.sum(lnq_beta) + w_x + w_z + logq_membership + np.sum(x + z) 
            
    
# Loader and Runner helper functions -------------------------------------------------------------------------------    

def gen_synth_data():

    eta = np.zeros((nclasses, nclasses, nclusters))
    beta = np.zeros((nclasses, 1, nclusters))
    for j in range(nclasses):
        for l in range(nclasses):
            a = phi0[j] * gamma0[j, l]
            b = np.sum(phi0[j] * gamma0[j, :]) - a
            eta[j, l, :] = beta_dist.rvs(a, b, size=nclusters)
    
        beta[j, 0, :] = gamma.rvs(a0[j], scale=b0[j], size=nclusters)

    #choose cluster membership
    r = np.zeros(nworkers, dtype=int)
    cluster_counts = np.zeros(nclusters)
    cluster_counts[0] += 1 # the first worker is always in cluster 0
    for k in range(1, nworkers):
        p_new_cluster = conc_prior_alpha / (conc_prior_alpha + k - 1)
        new_cluster = bernoulli.rvs(p_new_cluster)
        if new_cluster:
            r[k] = np.max(r) + 1
        else:
            probs = cluster_counts / (conc_prior_alpha + k - 1)
            assigned = False
            for cl in range(nclusters):
                assigned = bernoulli.rvs(probs[cl])
                if assigned:
                    r[k] = cl
                    break
            
        cluster_counts[r[k]] += 1
    
    print("Cluster memberships: ")
    print(r)
    
    pi = np.zeros((nclasses, nclasses, nworkers))
    for j in range(nclasses):
        for l in range(nclasses):
            a = beta[j, 0, r] * eta[j, l, r]
            b = np.sum(beta[j, :, r] * eta[j, :, r], axis=1) - a
            pi[j, l, :] = beta_dist.rvs(a, b, size=nworkers)
    
    kappa = np.zeros(nclasses)
    t = np.zeros(N, dtype=int) - 1
    for j in range(nclasses):
        kappa[j] = beta_dist.rvs(nu0[j], np.sum(nu0) - nu0[j])
        
    for j in range(nclasses):
        notset = t == -1
        if j == nclasses - 1:
            t[notset] = j
        else:
            thisclass = bernoulli.rvs(kappa[j] / np.sum(kappa[j:]), size=N)
            t[notset & thisclass.astype(bool)] = j
    
    C = np.zeros((N, nworkers)) - 1
    for k in range(nworkers):
        for l in range(nclasses):
            notset = C[:, k] == -1
            if l == nclasses - 1:
                C[notset, k] = l
            else:
                thisclass = bernoulli.rvs(pi[t, l, k] / (np.sum(pi[t, l:, k], axis=1)) )
                C[thisclass.astype(bool) & notset, k] = l
    return t, C, pi, kappa, eta, beta, r

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv)>1:
        configFile = sys.argv[1]
    else:
        configFile = './config/my_project.py'        
#     pT, combiner = ibcc.load_and_run_ibcc(configFile, ibcc_class=HCBCC)
#     pT2, combiner2 = ibcc.load_and_run_ibcc(configFile, ibcc_class=CBCC)
#     pTbase, combinerbase = ibcc.load_and_run_ibcc(configFile, ibcc_class=ibcc.IBCC)    

    N = 500
    nclasses = 3
    nworkers = 200
    nclusters = nworkers
    conc_prior_alpha = 0.1
    # generate dirichlet parameters for each cluster
    phi0 = np.zeros(nclasses) + 3
    gamma0 = np.zeros((nclasses, nclasses)) + 0.3
    gamma0[np.arange(nclasses), np.arange(nclasses)] = 0.7
    a0 = np.zeros(nclasses) + 30
    b0 = np.zeros(nclasses) + 2
    nu0 = np.zeros(nclasses) + 100

    t, C, pi, kappa, eta, beta, r = gen_synth_data()
    
    tablesize = C.shape[0] * C.shape[1]
    sparsity = 0.975
    acc_results = []
    while(sparsity > 0.825):
        availableidxs = np.random.choice(np.arange(tablesize), int(np.round(sparsity * tablesize)))
        availableidxs = np.unravel_index(availableidxs, dims=C.shape)
        Csparse = np.zeros(C.shape) - 1
        Csparse[availableidxs] = C[availableidxs]
        
        hcbcc_combiner = HCBCC(nclasses, nclasses, phi0, gamma0, nu0, a0, b0, C.shape[1], uselowerbound=True)
        hcbcc_combiner.uselowerbound = True        
        pT = hcbcc_combiner.combine_classifications(Csparse, table_format=True)
        
        sparsity -= 0.025
    
        predictedclass = np.argmax(pT, axis=1)
    
        from sklearn.metrics import accuracy_score
    
        acc = accuracy_score(t, predictedclass)
        print("accuracy: %f" % acc)
        acc_results.append(acc)
          
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(acc_results)
    
#     print hcbcc_combiner.r
#     
#     # test the similarities between inferred and ground truth of pi
#     hellinger = np.zeros((nclasses, nworkers))
#     abs_diff = np.zeros((nclasses, nworkers))
#     for k in range(nworkers):
#         for j in range(nclasses):
#             #print " Expected pi for cluster %i" % cl
#             pi_true = pi[j, :, k] #eta[j, :, r[k]] * beta
#             pi_inferred = (hcbcc_combiner.eta[j, :, :] * hcbcc_combiner.beta[j, :, :]).dot(hcbcc_combiner.r[k, :].T) 
#             pi_inferred = pi_inferred / np.sum(pi_inferred, axis=0)
#                             
#             hellinger[j, k] = 1.0 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(pi_true) - np.sqrt(pi_inferred))**2))
#             print "hellinger distance for class %i and worker %i: %f" % (j, k, hellinger[j,k])
#             
#             abs_diff[j, k] = np.abs(pi_true[1] - pi_inferred[1])
#             print "Absolut difference (exptect error rate) for class %i and worker %i: %f" % (j, k, abs_diff[j, k])