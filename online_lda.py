from scipy.stats import dirichlet, multinomial, expon
from collections import defaultdict
from math import pow
import numpy as np
import copy

from helpers import *


class OnlineLDA:
        def __init__(self, n_topics, n_documents, words_per_doc, vocab_size, alpha, eta, words_by_doc, tau=128.0, kappa=0.7):
            np.random.seed(0)
            
            self.K = n_topics
            self.D = n_documents
            self.N = words_per_doc
            self.V = vocab_size
            self.alpha = alpha
            self.eta = eta
            self.tau = tau
            self.kappa = kappa
            
            # Initialize variational parameters
            self.lamb = []  # Parameter for words distribution by topic (qBeta ~ Dirichlet(lamb)) 
            exp_parameter = float(self.K * self.V) / (self.D * self.N)
            for k in xrange(self.K):
                self.lamb.append(expon(scale=exp_parameter).rvs(size=self.V) + self.eta) # Suggested in Hoffman et al. (2013)
            self.lamb = np.array(self.lamb)
            self.gamma = np.ones((self.D, self.K)) # Parameter for topics assignments by doc (qTheta ~ Dirichlet(gamma))
            self.phi = np.zeros((self.D, self.N, self.K))
            
            self.words_by_doc = words_by_doc
        
        def fit(self):
            #Inicializo lambda como sugiere el paper, pie de pagina 1328
            docs = self.words_by_doc.keys()
            for t in xrange(15000):
                #sample uniformly a document
                # TODO: use mini-batches
                print 'Iteracion:', t
                d = np.random.choice(docs)
                print 'Documento:', d
                ro = pow((t + self.tau), (-self.kappa))
                print 'Ro:', ro
                self.e_step(d)
                self.m_step(d, ro)
        
        def e_step(self, d):
            words = self.words_by_doc[d]
            
            self.gamma[d] = np.random.gamma(shape=self.K, scale=1./self.K, size=self.K)
            expected_log_theta = dirich_log_expectation(self.gamma[d]) # shape (K, 1)
            exponential_elog_theta = np.exp(expected_log_theta)
            
            # The E-Step doesn't updates lambda parameter.
            expected_log_beta = dirich_log_expectation(self.lamb)[:, words] # shape (K, N)
            exponential_elog_beta = np.exp(expected_log_beta)
            #print 'expected_log_beta:', np.mean(expected_log_beta), np.std(expected_log_beta)
            #print 'lambda:', np.mean(self.lamb)
            
            phinorm = np.dot(exponential_elog_theta.T, exponential_elog_beta) + 1e-100 # Cada componente del vector es algo asi como la probabilidad de ver la palabra 
                                                                                       # w, sin informacion sobre el topico

            mean_change = 1.0 # Variable for determine if convergence has been achieved
            for _ in xrange(100):
                prev_gamma = copy.deepcopy(self.gamma[d])
                
                # I am leveraging vectorization over the topics and the words
                # Updates phi[d]
                self.phi[d] = np.exp(expected_log_theta + expected_log_beta).T / phinorm.T # shape (N, K)
            
                # Updates gamma[d]
                self.gamma[d] = self.alpha + np.sum(self.phi[d], axis=0)
                mean_change = np.mean(abs(self.gamma[d] - prev_gamma))
                #print 'Mean change:', mean_change
                
                expected_log_theta = dirich_log_expectation(self.gamma[d]) # shape (K, 1)
                exponential_elog_theta = np.exp(expected_log_theta)
                phinorm = np.dot(exponential_elog_theta.T, exponential_elog_beta) + 1e-100
                if mean_change < 0.001:
                    break
                
        def m_step(self, d, ro):
            indicator_words = one_hot_encoding(self.words_by_doc[d], self.V)
            intermediate_lambda = self.eta + self.D * np.dot(self.phi[d].T, indicator_words)
            self.lamb = (1 - ro) * self.lamb + ro * intermediate_lambda