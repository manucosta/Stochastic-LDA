import copy
import logging
from tqdm import tqdm
from math import pow

from helpers import *

logger = logging.getLogger(__name__)


class OnlineLDA:
    def __init__(self, n_topics, n_documents, words_per_doc, vocab_size, alpha, eta, words_by_doc, tau=128.0, kappa=0.7):
        """

        :param n_topics: the number of topics/clusters that the model will learn
        :param n_documents: the number of documents in the corpus
        :param words_per_doc: the fixed number of words per document
        :param vocab_size: the number of distinct words in the vocabulary
        :param alpha: parameter for prior distribution of theta (ie, topic proportions per document)
        :param eta: parameter for prior distribution
        :param words_by_doc: a dictionary mapping document_ids to lists of word_id
        :param tau: the delay (down-weights early iterations). It must be a postive real number or zero
        :param kappa: the forgetting rate. It must fall in the range (0.5, 1] in order to guarantee convergence
        """
        np.random.seed(0)

        # Hyperparameters
        self.K = n_topics
        self.D = n_documents
        self.N = words_per_doc
        self.V = vocab_size

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
        for t in tqdm(xrange(15000)):
            logger.debug('Iteration: {}'.format(t))
            # Sample uniformly a document
            d = np.random.choice(docs)
            logger.debug('Document: {}'.format(d))
            # Compute the learning rate, ro
            ro = pow((t + self.tau), (-self.kappa))
            logger.debug('Ro: {}'.format(ro))
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
        # print 'expected_log_beta:', np.mean(expected_log_beta), np.std(expected_log_beta)
        # print 'lambda:', np.mean(self.lamb)

        phinorm = np.dot(exponential_elog_theta.T, exponential_elog_beta) + 1e-100

        for _ in xrange(100):
            prev_gamma = copy.deepcopy(self.gamma[d])

            # I am leveraging vectorization over the topics and the words
            # Updates phi[d]
            self.phi[d] = np.exp(expected_log_theta + expected_log_beta).T / phinorm.T # shape (N, K)

            # Updates gamma[d]
            self.gamma[d] = self.alpha + np.sum(self.phi[d], axis=0)
            mean_change = np.mean(abs(self.gamma[d] - prev_gamma))
            # print 'Mean change:', mean_change

            expected_log_theta = dirich_log_expectation(self.gamma[d]) # shape (K, 1)
            exponential_elog_theta = np.exp(expected_log_theta)
            phinorm = np.dot(exponential_elog_theta.T, exponential_elog_beta) + 1e-100
            if mean_change < 0.001:
                break

    def m_step(self, d, ro):
        indicator_words = one_hot_encoding(self.words_by_doc[d], self.V)
        intermediate_lambda = self.eta + self.D * np.dot(self.phi[d].T, indicator_words)
        self.lamb = (1 - ro) * self.lamb + ro * intermediate_lambda
