import copy
import logging
from math import pow

from helpers import *

logger = logging.getLogger(__name__)


class MiniBatchLDA:

    def __init__(self, n_topics, n_documents, words_per_doc, vocab_size, alpha, eta, words_by_doc,
                 minibatch_size=64, tau=1.0, kappa=0.8):
        """

        :param n_topics: the number of topics/clusters that the model will learn
        :param n_documents: the number of documents in the corpus
        :param words_per_doc: the fixed number of words per document
        :param vocab_size: the number of distinct words in the vocabulary
        :param alpha: parameter for prior distribution of theta (ie, topic proportions per document)
        :param eta: parameter for prior distribution
        :param words_by_doc: a dictionary mapping document_ids to lists of word_id
        :param minibatch_size: the size of the mini-batch. By default, its value is 64
        :param tau: the delay (down-weights early iterations). It must be a postive real number or zero
        :param kappa: the forgetting rate. It must fall in the range (0.5, 1] in order to guarantee convergence
        """

        # TODO: allow a variable length of words per doc
        np.random.seed(0)

        # Hyperparameters
        self.K = n_topics
        self.D = n_documents
        self.N = words_per_doc
        self.V = vocab_size
        self.minibatch_size = minibatch_size

        # Priors
        self.alpha = alpha
        self.eta = eta

        # Learning rate parameters
        self.tau = tau
        self.kappa = kappa

        # Initialize variational parameters
        # Lambda: Parameters for words distribution by topic (qBeta ~ Dirichlet(lamb))
        # Initialize lambda as suggested by Hoffman et al. (2013) at the footnote in page 1328
        exp_parameter = float(self.K * self.V) / (self.D * self.N)
        self.lamb = np.random.exponential(scale=exp_parameter, size=(self.K, self.V)) + self.eta  # (K, V)
        # Phi: Parameters for topic assignment of the nth word in the dth document (qZ ~ Multinomial(phi))
        self.phi = np.zeros((self.D, self.N, self.K))   # (D, N, K)
        # Gamma: Parameters for topic proportions in each document (qTheta ~ Dirichlet(gamma))
        self.gamma = np.ones((self.D, self.K))  # (D, K)

        self.words_by_doc = words_by_doc

    def fit(self):
        docs = self.words_by_doc.keys()
        for t in xrange(1000):
            logger.info('Iteration: {}'.format(t))
            # Sample uniformly a minibatch of documents allowing replacement
            ds = np.random.choice(docs, size=self.minibatch_size)
            logger.debug('Picked documents: {}'.format(ds))
            # Compute the learning rate, ro
            ro = pow((t + self.tau), (-self.kappa))
            logger.info('Ro: {}'.format(ro))
            self.e_step(ds)
            self.m_step(ds, ro)

    def e_step(self, ds):
        gamma = np.random.gamma(shape=self.K, scale=1./self.K, size=(self.K, len(ds)))
        logger.debug('gamma shape: {}'.format(gamma.shape))
        expected_log_theta = dirich_log_expectation(gamma) # shape (K, mini_batches)
        logger.debug('expected_log_theta shape: {}'.format(expected_log_theta.shape))
        exponential_elog_theta = np.exp(expected_log_theta)
        logger.debug('exponential_elog_theta shape: {}'.format(exponential_elog_theta.shape))

        for idx, d in enumerate(ds):
            words = self.words_by_doc[d]

            expected_log_thetad = np.reshape(expected_log_theta[:, idx], (self.K, 1))
            exponenetial_elog_thetad = exponential_elog_theta[:, idx]

            # The E-Step doesn't updates lambda parameter.
            expected_log_betad = dirich_log_expectation(self.lamb)[:, words]    # shape (K, N)
            exponential_elog_betad = np.exp(expected_log_betad)

            phinorm = np.dot(exponenetial_elog_thetad.T, exponential_elog_betad) + 1e-100

            for _ in xrange(100):
                prev_gamma = copy.deepcopy(gamma[:, idx])

                # Updates phi[d]
                self.phi[d] = np.exp(expected_log_thetad + expected_log_betad).T / np.reshape(phinorm.T, (self.N, 1))   # shape (N, K)

                # Updates gamma[d]
                gamma[:, idx] = self.alpha + np.sum(self.phi[d], axis=0)
                mean_change = np.mean(abs(gamma[:, idx] - prev_gamma))
                # print 'Mean change:', mean_change

                expected_log_thetad = dirich_log_expectation(gamma[:, idx])     # shape (K, 1)
                exponential_elog_thetad = np.exp(expected_log_thetad)
                phinorm = np.dot(exponential_elog_thetad.T, exponential_elog_betad) + 1e-100

                if mean_change < 0.001:
                    break
            self.gamma[d] = gamma[:, idx].ravel()

    def m_step(self, ds, ro):
        intermediate_lambda = 0.0
        for d in ds:
            indicator_words = one_hot_encoding(self.words_by_doc[d], self.V)
            intermediate_lambda += self.eta + self.D * np.dot(self.phi[d].T, indicator_words)
        self.lamb = (1 - ro) * self.lamb + ro * intermediate_lambda / self.minibatch_size
