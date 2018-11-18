from scipy.stats import dirichlet, multinomial, expon
from scipy.special import digamma
from collections import defaultdict
from math import pow
import numpy as np
import copy

from helpers import *


class MiniBatchLDA:
        
        def __init__(self, n_topics, n_documents, words_per_doc, vocab_size, alpha, eta, words_by_doc, minibatch_size, tau=128.0, kappa=0.7):
            np.random.seed(0)
            
            self.K = n_topics
            self.D = n_documents
            self.N = words_per_doc
            self.V = vocab_size
            self.alpha = alpha
            self.eta = eta
            self.tau = tau
            self.kappa = kappa
            self.minibatch_size = minibatch_size
            
            # Initialize variational parameters
            self.lamb = []  # Parameter for words distribution by topic (qBeta ~ Dirichlet(lamb)) 
            exp_parameter = float(self.K * self.V) / (self.D * self.N)
            for k in xrange(self.K):
                self.lamb.append(expon(scale=exp_parameter).rvs(size=self.V) + self.eta) # Suggested in Hoffman et al. (2013)
            self.lamb = np.array(self.lamb)
            self.phi = np.zeros((self.D, self.N, self.K))
            
            self.words_by_doc = words_by_doc
        
        def fit(self):
            #Inicializo lambda como sugiere el paper, pie de pagina 1328
            docs = self.words_by_doc.keys()
            for t in xrange(500):
                #sample uniformly a document
                # TODO: use mini-batches
                print 'Iteracion:', t
                ds = np.random.choice(docs, size=self.minibatch_size)
                print 'Documentos:', ds
                ro = pow((t + self.tau), (-self.kappa))
                print 'Ro:', ro
                self.e_step(ds)
                self.m_step(ds, ro)
        
        def e_step(self, ds):
            gamma = np.random.gamma(shape=self.K, scale=1./self.K, size=(self.K, len(ds)))
            #print gamma.shape
            expected_log_theta = dirich_log_expectation(gamma) # shape (K, mini_batches)
            #print expected_log_theta.shape
            exponential_elog_theta = np.exp(expected_log_theta)
            #print exponential_elog_theta.shape
            
            for idx, d in enumerate(ds):
                words = self.words_by_doc[d]
                
                expected_log_thetad = np.reshape(expected_log_theta[:, idx], (self.K, 1))
                exponenetial_elog_thetad = exponential_elog_theta[:, idx]

                # The E-Step doesn't updates lambda parameter.
                expected_log_betad = dirich_log_expectation(self.lamb)[:, words] # shape (K, N)
                exponential_elog_betad = np.exp(expected_log_betad)

                phinorm = np.dot(exponenetial_elog_thetad.T, exponential_elog_betad) + 1e-100 # Cada componente del vector es algo asi como la probabilidad de ver la palabra 
                                                                                           # w, sin informacion sobre el topico

                mean_change = 1.0 # Variable for determine if convergence has been achieved
                for _ in xrange(100):
                    prev_gamma = copy.deepcopy(gamma[:, idx])

                    # I am leveraging vectorization over the topics and the words
                    # Updates phi[d]
                    self.phi[d] = np.exp(expected_log_thetad + expected_log_betad).T / np.reshape(phinorm.T, (self.N, 1)) # shape (N, K)

                    # Updates gamma[d]
                    gamma[:, idx] = self.alpha + np.sum(self.phi[d], axis=0)
                    mean_change = np.mean(abs(gamma[:, idx] - prev_gamma))
                    #print 'Mean change:', mean_change

                    expected_log_thetad = dirich_log_expectation(gamma[:, idx]) # shape (K, 1)
                    exponential_elog_thetad = np.exp(expected_log_thetad)
                    phinorm = np.dot(exponential_elog_thetad.T, exponential_elog_betad) + 1e-100
                    
                    if mean_change < 0.001:
                        break
                
        def m_step(self, ds, ro):
            # TODO: use the account of words per document
            intermediate_lambda = 0.0
            for d in ds:
                indicator_words = one_hot_encoding(self.words_by_doc[d], self.V)
                intermediate_lambda += self.eta + self.D * np.dot(self.phi[d].T, indicator_words)
            self.lamb = (1 - ro) * self.lamb + ro * intermediate_lambda / self.minibatch_size