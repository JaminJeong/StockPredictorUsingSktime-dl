# -*- coding: utf-8 -*-
""" WEASEL classifier
dictionary based classifier based on SFA transform, BOSS and linear regression.
"""

__author__ = "Patrick Schäfer"
__all__ = ["WEASEL"]

import math

import numpy as np
from numba import njit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sktime.classification.base import BaseClassifier
from sktime.transformers.series_as_features.dictionary_based import SFA
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y


# from sklearn.feature_selection import chi2

# from numba.typed import Dict


class WEASEL(BaseClassifier):
    """Word ExtrAction for time SEries cLassification (WEASEL)

    WEASEL: implementation of WEASEL from Schäfer:
    @inproceedings{schafer2017fast,
      title={Fast and Accurate Time Series Classification with WEASEL},
      author={Sch{\"a}fer, Patrick and Leser, Ulf},
      booktitle={Proceedings of the 2017 ACM on Conference on Information and
                 Knowledge Management},
      pages={637--646},
      year={2017}
    }
    # Overview: Input n series length m
    # WEASEL is a dictionary classifier that builds a bag-of-patterns using SFA
    # for different window lengths and learns a logistic regression classifier
    # on this bag.
    #
    # There are these primary parameters:
    #         alphabet_size: alphabet size
    #         chi2-threshold: used for feature selection to select best words
    #         anova: select best l/2 fourier coefficients other than first ones
    #         bigrams: using bigrams of SFA words
    #         binning_strategy: the binning strategy used to discretise into
    #                           SFA words.
    #
    # WEASEL slides a window length w along the series. The w length window
    # is shortened to an l length word through taking a Fourier transform and
    # keeping the best l/2 complex coefficients using an anova one-sided
    # test. These l coefficients are then discretised into alpha possible
    # symbols, to form a word of length l. A histogram of words for each
    # series is formed and stored.
    # For each window-length a bag is created and all words are joint into
    # one bag-of-patterns. Words from different window-lengths are
    # discriminated by different prefixes.
    #
    # fit involves training a logistic regression classifier on the single
    # bag-of-patterns.
    #
    # predict uses the logistic regression classifier

    # For the Java version, see
    # https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    # /tsml/classifiers/dictionary_based/WEASEL.java


    Parameters
    ----------
    anova:               boolean, default = True
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.
        Only applicable if labels are given

    bigrams:             boolean, default = True
        whether to create bigrams of SFA words

    binning_strategy:    {"equi-depth", "equi-width", "information-gain"},
                         default="information-gain"
        The binning method used to derive the breakpoints.

    window_inc:          int, default = 4
        WEASEL create a BoP model for each window sizes. This is the
        increment used to determine the next window size.

    chi2_threshold:      int, default = -1 (disabled by default)
        Feature selection is applied based on the chi-squared test.
        This is the threshold to use for chi-squared test on bag-of-words
        (higher means more strict). Negative values indicate that the test
        should not be performed.

    random_state:        int or None,
        Seed for random, integer

    Attributes
    ----------


    """

    def __init__(
        self,
        anova=True,
        bigrams=True,
        binning_strategy="information-gain",
        window_inc=4,
        chi2_threshold=-1,  # disabled by default
        random_state=None,
    ):

        # currently other values than 4 are not supported.
        self.alphabet_size = 4

        # feature selection is applied based on the chi-squared test.
        self.chi2_threshold = chi2_threshold

        self.anova = anova

        self.norm_options = [True, False]
        self.word_lengths = [4, 6]

        self.bigrams = bigrams
        self.binning_strategy = binning_strategy
        self.random_state = random_state

        self.min_window = 4
        self.max_window = 350

        self.window_inc = window_inc
        self.highest_bit = -1
        self.window_sizes = []

        self.series_length = 0
        self.n_instances = 0

        self.SFA_transformers = []
        self.clf = None
        self.best_word_length = -1

        super(WEASEL, self).__init__()

    def fit(self, X, y):
        """Build a WEASEL classifiers from the training set (X, y),

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        # Window length parameter space dependent on series length
        self.n_instances, self.series_length = X.shape[0], X.shape[-1]

        win_inc = self.compute_window_inc()

        self.max_window = int(min(self.series_length, self.max_window))
        self.window_sizes = list(range(self.min_window, self.max_window, win_inc))

        self.highest_bit = (math.ceil(math.log2(self.max_window))) + 1
        rng = check_random_state(self.random_state)

        all_words = [dict() for x in range(len(X))]

        for window_size in self.window_sizes:

            transformer = SFA(
                word_length=rng.choice(self.word_lengths),
                alphabet_size=self.alphabet_size,
                window_size=window_size,
                norm=rng.choice(self.norm_options),
                anova=self.anova,
                # levels=rng.choice([1, 2, 3]),
                binning_method=self.binning_strategy,
                bigrams=self.bigrams,
                remove_repeat_words=False,
                lower_bounding=False,
                save_words=False,
            )

            sfa_words = transformer.fit_transform(X, y)

            self.SFA_transformers.append(transformer)
            bag = sfa_words[0]  # .iloc[:, 0]

            # chi-squared test to keep only relevant features
            relevant_features = {}
            apply_chi_squared = self.chi2_threshold > 0
            if apply_chi_squared:
                bag_vec = DictVectorizer(sparse=False).fit_transform(bag)
                chi2_statistics, p = chi2(bag_vec, y)
                relevant_features = np.where(chi2_statistics >= self.chi2_threshold)[0]

            # merging bag-of-patterns of different window_sizes
            # to single bag-of-patterns with prefix indicating
            # the used window-length
            for j in range(len(bag)):
                for (key, value) in bag[j].items():
                    # chi-squared test
                    if (not apply_chi_squared) or (key in relevant_features):
                        # append the prefices to the words to
                        # distinguish between window-sizes
                        if isinstance(key, tuple):
                            word = (
                                ((key[0] << self.highest_bit) | key[1]) << 3
                            ) | window_size
                        else:
                            # word = ((key << self.highest_bit) << 3) \
                            #        | window_size
                            word = WEASEL.shift_left(key, self.highest_bit, window_size)

                        all_words[j][word] = value

        self.clf = make_pipeline(
            DictVectorizer(sparse=False),
            StandardScaler(with_mean=True, copy=False),
            LogisticRegression(
                max_iter=5000,
                solver="liblinear",
                dual=True,
                # class_weight="balanced",
                penalty="l2",
                random_state=self.random_state,
            ),
        )

        self.clf.fit(all_words, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        bag = self._transform_words(X)
        return self.clf.predict(bag)

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        bag = self._transform_words(X)
        return self.clf.predict_proba(bag)

    def _transform_words(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        bag_all_words = [dict() for _ in range(len(X))]
        for i, window_size in enumerate(self.window_sizes):

            # SFA transform
            sfa_words = self.SFA_transformers[i].transform(X)
            bag = sfa_words[0]  # .iloc[:, 0]

            # merging bag-of-patterns of different window_sizes
            # to single bag-of-patterns with prefix indicating
            # the used window-length
            for j in range(len(bag)):
                for (key, value) in bag[j].items():
                    # append the prefices to the words to distinguish
                    # between window-sizes
                    if isinstance(key, tuple):
                        word = (
                            ((key[0] << self.highest_bit) | key[1]) << 3
                        ) | window_size
                    else:
                        # word = ((key << self.highest_bit) << 3) | window_size
                        word = WEASEL.shift_left(key, self.highest_bit, window_size)

                    bag_all_words[j][word] = value

        return bag_all_words

    def compute_window_inc(self):
        win_inc = self.window_inc
        if self.series_length < 50:
            win_inc = 1  # less than 50 is ok time-wise
        elif self.series_length < 100:
            win_inc = min(self.window_inc, 2)  # less than 50 is ok time-wise
        return win_inc

    @staticmethod
    @njit(fastmath=True, cache=True)
    def shift_left(key, highest_bit, window_size):
        return ((key << highest_bit) << 3) | window_size
