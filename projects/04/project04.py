import os
import pandas as pd
import numpy as np
import requests
import time
import re


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    book = requests.get(url)
    book_str = book.text.replace('\r\n', '\n')
    s = re.search('(\*\*\* START.*)', book_str)  # START
    e = re.search('(\*\*\* END.*)', book_str)  # END

    return book_str[s.end(0):e.start(0)]


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    add_start_end = re.sub('(\n\n)+', ' \x03 \x02 ', book_string)
    remove_whitespace = re.sub('\s+', ' ', add_start_end)
    beginning = re.sub('\A', '\x02 ', remove_whitespace)
    cleaned = re.sub('\Z', '\x03', beginning)

    return re.findall('[\w]+|\\x02|\\x03|[^\w\s]', cleaned)


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        token_series = pd.Series(tokens).value_counts()

        return token_series.apply(lambda x: 1 / len(token_series))

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        P = 1
        for word in words:
            if word not in self.mdl.index:
                return 0
            P = P * self.mdl[word]
        return P

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """

        return ' '.join(list(self.mdl.sample(M, replace=True,
                                             weights=self.mdl).index)).strip()


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):

    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """

        return pd.Series(tokens).value_counts() / pd.Series(tokens). \
            value_counts().sum()

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        P = 1
        for word in words:
            if word not in self.mdl.index:
                return 0
            P = P * self.mdl[word]

        return P

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """

        return ' '.join(list(self.mdl.sample(M, replace=True,
                                             weights=self.mdl).index)).strip()


# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):

    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N - 1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        Ngram_list = []

        for i in range(len(tokens)):
            if tokens[0] == '\x02':
                Ngram = []
                if i < self.N - 1:
                    continue
                for j in range(len(tokens)):
                    if i - self.N < j <= i:
                        Ngram.append(tokens[j])

                Ngram_list.append(tuple(Ngram))
                if tokens[i] == '\x03':
                    break

        return Ngram_list

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        def n_1gram(tup):
            minus_one = []
            for i in range(len(tup) - 1):
                minus_one.append(tup[i])
            return tuple(minus_one)

        ngram = pd.Series(ngrams)
        n1gram = pd.Series(ngrams).apply(n_1gram)
        df = pd.DataFrame({'ngram': ngram, 'n1gram': n1gram})

        # ngram counts C(w_1, ..., w_n)
        n_counts = df.groupby('ngram')['ngram'].transform('count')

        # n-1 gram counts C(w_1, ..., w_(n-1))
        n1_counts = df.groupby('n1gram')['n1gram'].transform('count')

        # Create the conditional probabilities
        cond_probs = n_counts / n1_counts

        # Put it all together
        df['prob'] = cond_probs

        return df

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        words = words[::-1]
        model = self
        N = self.N
        P = 1.0
        for i in range(len(words)):
            gram = []
            mdl = model.mdl
            if i > self.N - 1:  # N-1
                N = N - 1  # N= N-1
                model = self.prev_mdl
                mdl = model.mdl
                if N == 1:
                    mdl = pd.DataFrame(mdl).rename(
                        columns={0: 'prob'}).reset_index().rename(
                        columns={'index': 'ngram'})
                    for j in range(N):
                        gram.append(words[i + j])

                    P = P * mdl[(mdl['ngram'] == tuple(gram)[0])]['prob'].max()
                else:
                    for j in range(N):
                        gram.append(words[i + j])
                    P = P * mdl[(mdl['ngram'] == tuple(gram)[::-1]) & (
                                mdl['n1gram'] == tuple(gram)[::-1][:-1])][
                        'prob'].max()
            else:
                for j in range(N):  # N
                    gram.append(words[i + j])  # N
                P = P * mdl[(mdl['ngram'] == tuple(gram)[::-1]) & (
                            mdl['n1gram'] == tuple(gram)[::-1][:-1])][
                    'prob'].max()

        return P

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """

        # Use a helper function to generate sample tokens of length `length`
        ...

        # Transform the tokens to strings
        ...

        return ' '.join(list(self.mdl.sample(M, replace=True,
                                             weights=self.mdl).index)).strip()


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """

    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
