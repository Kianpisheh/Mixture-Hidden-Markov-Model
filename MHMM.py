from hmmlearn import hmm
import numpy as np
from scipy.stats import multivariate_normal

class hmm2:

    def __init__(self, n_states):
        self.n_states = n_states
        self.trans_mat =  np.ones((n_states,n_states)) / n_states
        self.means = None
        self.cov = None
        self.pi = np.ones(n_states) * (1.0 / n_states)
        self._hidden_states = np.zeros(n_states)
        self._alpha = np.zeros(n_states)
        self._beta = np.zeros(n_states)
        self._model_type = None

    def set_emmision_model(self, model_type, d):
        self._model_type = model_type
        if model_type == "gaussian":
            self.d = d
            self.means = np.random.rand(self.n_states, 1, d)
            self.cov = np.tile(np.identity(d), (self.n_states, 1, 1))

    def _emission_prob(self, x, z):
        """
        Args:
            z: hidden variable. scalar
            x: observation at time t. size: (d,)
        """
        if self._model_type == "gaussian":
            return multivariate_normal.pdf(x, self.means[z], self.cov[z])

    def _forward(self, X):
        """
        Args:
            X: observation sequence. size: (d,T)
               d -> feature dim, T -> seqeunce length
        """
        T = X.shape[1]
        self._log_alpha = np.ones((self.n_states, T))
        for z_t in range(self.n_states):
            self._log_alpha[z_t, 0] = self._compute_alpha(X[:,0], z_t)
        for t in range(1, T):
            for z_t in range(self.n_states):
                self._alpha[z_t,t] = self._compute_alpha(X[:,t], z_t, self._alpha[z_t,t-1]) 
        return self._alpha

    def _backward(self, X):
        """
        Args:
            X: observation sequence. size: (d,T)
               d -> feature dim, T -> seqeunce length
        """
        T = X.shape[1]
        self._beta = np.zeros((self.n_states, T))
        self._beta[:,-1] = self._compute_beta(X[:,-1], 0)
        for t in reversed(range(0, T-1)):
            for z_t in range(self.n_states):
                self._beta[z_t,t] = self._compute_beta(X[:,t], z_t, self._beta[z_t,t+1])
        return self._beta

    def _compute_alpha(self, x_t, z_t, prev_alpha=None):
        """
        alpha_t(s): P(X1:t, z_t=s)
        Args:
            x_t: observation at time t; size (d,)
            z_t: hidden state value at time t; scalar
            prev_alpha: value of alpha at time (t-1) for each hidden variable.
                size: (n_states,)
        """
        if prev_alpha is None:
            return np.log(self.pi[z_t]) + np.log(self._emission_prob(x_t, z_t))
        else:
            return np.sum(prev_alpha * self.trans_mat[:,z_t]) * self._emission_prob(x_t, z_t)

    def _compute_beta(self, X_t, z_t, next_beta=None):
        """
        beta_t(s): P(X_t+1:T | z_t=s)
        Args:
            X_t: observation at time t; numpy array of size (d,)
            z_t: hidden state value at time t; scalar
            next_beta: next value of beta (t+1) for each hidden variable.
                        numpy array of size (n_states,)
        """
        if next_beta is None:
            return 1
        else:
            emission_probs = np.array([self._emission_prob(X_t, s) for s in range(self.n_states)])
        return np.sum(next_beta * self.trans_mat[z_t,:] * emission_probs)


    def hidden_vars_dist(self):
        prob_za_X = self._beta * self._alpha
        return prob_za_X / np.tile(prob_za_X.sum(axis=1).reshape(-1,1), (1, prob_za_X.shape[1]))
        
    def fit(self, X):
        """
        Args:
            X: observation sequence. size: (d,T)
               d -> feature dim, T -> seqeunce length
        """
        return self._em_step(X)


    def _m_step(self):
        pass

    # TODO: alpha nd beta normalization
    def _em_step(self, X_i):
        """
        E-step for sequence sample X_i
        Args:
            X: observation sequence. size: (d,T)
               d -> feature dim, T -> seqeunce length
        """
        for i in range(100):
            model = hmm.GaussianHMM(n_components=3, covariance_type="full")
            model.startprob_ = np.array([0.6, 0.3, 0.1])
            model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                        [0.3, 0.5, 0.2],
                                        [0.3, 0.3, 0.4]])
            model.means_ = np.array([[0.0], [3.0], [5.0]])
            model.covars_ = np.tile(np.identity(1), (3, 1, 1))
            X_i, Z = model.sample(100)
            X_i = X_i.reshape(1,-1)
            T = X_i.shape[1]
            alpha = self._forward(X_i)
            print(alpha[:,:3])
            beta = self._backward(X_i)
            p_X = np.sum(self._alpha[:,-1])   # P^(X)

            p_zt_X = self._alpha * self._beta               # P^(zt,X)
            gamma = p_zt_X / np.sum(p_zt_X, axis=0)         # P^(zt|X)
            eta = np.zeros((T,self.n_states,self.n_states)) # P^(zt,zt-1|X)
            for t in range(1, T):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        eta[t,i,j] = alpha[i,t-1] * self.trans_mat[i,j] * \
                            self._emission_prob(X_i[:,t], j) * beta[j,t]
            eta = eta / p_X
                
            # parameter updates
            self.pi = gamma[:,0] / np.sum(gamma[:,0])
            self.trans_mat = np.sum(eta[1:,:,:], axis=0) / np.sum(np.sum(eta, axis=2)[1:,:], axis=0)
            self.means = (X_i.dot(gamma.T) / np.sum(gamma, axis=1)).reshape(self.n_states,1,self.d)
            for i in range(self.n_states):
                for t in range(T):
                    a = X_i[:,t] - (self.means[i,0,:])
                    self.cov[i,:,:] += gamma[i,t] * a.dot(a.T)
                self.cov[i,:,:] /= np.sum(gamma[i,:])

        return (self.pi, self.trans_mat, self.means, self.cov)

class MHMM:
    def __init__(self, n_clusters, n_states):
        self._n_clusters = n_clusters
        self._n_states = n_states
