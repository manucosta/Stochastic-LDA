import numpy as np
from scipy.special import digamma
from IPython.display import display, Markdown


def dirich_log_expectation(alpha):
    if len(alpha.shape) == 1:
        # Case gamma, of shape (K, )
        res = digamma(alpha) - digamma(np.sum(alpha, keepdims=True))
        return np.reshape(res, (res.shape[0], 1))
    else:
        # Case phi, of shape (K, V)
        return digamma(alpha) - digamma(np.sum(alpha, axis=1, keepdims=True))


def one_hot_encoding(words, length):
    """
    Arguments:
        words: list of length N, with the positions of the desired non-zero elements
        length: the length of each encoded vector
    Output:
        A matrix of shape (N, length), where each row is a 1-hot encoding
    """
    res = np.zeros((len(words), length))
    res[np.arange(res.shape[0]), words] = 1.0
    return res


def print_md(string, bold=False):
    if bold:
        string = '**' + string + '**'
    display(Markdown(string))


def print_words(ids, vocab):
    for id in ids:
        print '{id}, {word}'.format(id=id, word=vocab[id])


def print_top_k_topics_for_doc(lda, doc_id, k, words_by_doc, vocab):
    # Print the words of the document
    words = []
    for word_id in words_by_doc[doc_id]:
        words.append(vocab[word_id])
    print words

    # Print the main topics, with a sample of their main words
    topic_proportions_parameter = lda.gamma[doc_id]
    print topic_proportions_parameter
    topic_proportions = np.random.dirichlet(topic_proportions_parameter)
    top_k_topics = topic_proportions.argsort()[::-1][:k]
    for topic_id in top_k_topics:
        print_md('Topic {}'.format(topic_id), bold=True)
        np.random.seed(42)
        words_distribution = np.random.dirichlet(lda.lamb[topic_id])
        print_words(words_distribution.argsort()[::-1][:10], vocab)
