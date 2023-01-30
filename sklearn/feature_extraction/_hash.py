# Author: Lars Buitinck
# License: BSD 3 clause

from numbers import Integral
from bloom_filter2 import BloomFilter

from itertools import chain

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, TransformerMixin
from ._hashing_fast import transform as _hashing_transform
from ..utils._param_validation import Interval, StrOptions


def _iteritems(d):
    """Like d.iteritems, but accepts any collections.Mapping."""
    return d.iteritems() if hasattr(d, "iteritems") else d.items()


class FeatureHasher(TransformerMixin, BaseEstimator):
    """Implements feature hashing, aka the hashing trick.

    This class turns sequences of symbolic feature names (strings) into
    scipy.sparse matrices, using a hash function to compute the matrix column
    corresponding to a name. The hash function employed is the signed 32-bit
    version of Murmurhash3.

    Feature names of type byte string are used as-is. Unicode strings are
    converted to UTF-8 first, but no Unicode normalization is done.
    Feature values must be (finite) numbers.

    This class is a low-memory alternative to DictVectorizer and
    CountVectorizer, intended for large-scale (online) learning and situations
    where memory is tight, e.g. when running prediction code on embedded
    devices.

    Read more in the :ref:`User Guide <feature_hashing>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_features : int, default=2**20
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    input_type : str, default='dict'
        Choose a string from {'dict', 'pair', 'string'}.
        Either "dict" (the default) to accept dictionaries over
        (feature_name, value); "pair" to accept pairs of (feature_name, value);
        or "string" to accept single strings.
        feature_name should be a string, while value should be a number.
        In the case of "string", a value of 1 is implied.
        The feature_name is hashed to find the appropriate column for the
        feature. The value's sign might be flipped in the output (but see
        non_negative, below).
    dtype : numpy dtype, default=np.float64
        The type of feature values. Passed to scipy.sparse matrix constructors
        as the dtype argument. Do not set this to bool, np.boolean or any
        unsigned integer type.
    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.

        .. versionchanged:: 0.19
            ``alternate_sign`` replaces the now deprecated ``non_negative``
            parameter.

    See Also
    --------
    DictVectorizer : Vectorizes string-valued features using a hash table.
    sklearn.preprocessing.OneHotEncoder : Handles nominal/categorical features.

    Notes
    -----
    This estimator is :term:`stateless` and does not need to be fitted.
    However, we recommend to call :meth:`fit_transform` instead of
    :meth:`transform`, as parameter validation is only performed in
    :meth:`fit`.

    Examples
    --------
    >>> from sklearn.feature_extraction import FeatureHasher
    >>> h = FeatureHasher(n_features=10)
    >>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    >>> f = h.transform(D)
    >>> f.toarray()
    array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
           [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

    With `input_type="string"`, the input must be an iterable over iterables of
    strings:

    >>> h = FeatureHasher(n_features=8, input_type="string")
    >>> raw_X = [["dog", "cat", "snake"], ["snake", "dog"], ["cat", "bird"]]
    >>> f = h.transform(raw_X)
    >>> f.toarray()
    array([[ 0.,  0.,  0., -1.,  0., -1.,  0.,  1.],
           [ 0.,  0.,  0., -1.,  0., -1.,  0.,  0.],
           [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  1.]])
    """

    _parameter_constraints: dict = {
        "n_features": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # delegate to numpy
        "alternate_sign": ["boolean"],
    }

    def __init__(
        self,
        n_features=(2**20),
        *,
        input_type="dict",
        dtype=np.float64,
        alternate_sign=True,
    ):
        self.dtype = dtype
        self.input_type = input_type
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    def fit(self, X=None, y=None):
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        # repeat input validation for grid search (which calls set_params)
        self._validate_params()
        return self

    def transform(self, raw_X):
        """Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        raw_X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        raw_X = iter(raw_X)
        if self.input_type == "dict":
            raw_X = (_iteritems(d) for d in raw_X)
        elif self.input_type == "string":
            first_raw_X = next(raw_X)
            if isinstance(first_raw_X, str):
                raise ValueError(
                    "Samples can not be a single string. The input must be an iterable"
                    " over iterables of strings."
                )
            raw_X_ = chain([first_raw_X], raw_X)
            raw_X = (((f, 1) for f in x) for x in raw_X_)

        indices, indptr, values = _hashing_transform(
            raw_X, self.n_features, self.dtype, self.alternate_sign, seed=0
        )
        n_samples = indptr.shape[0] - 1

        if n_samples == 0:
            raise ValueError("Cannot vectorize empty sequence.")

        X = sp.csr_matrix(
            (values, indices, indptr),
            dtype=self.dtype,
            shape=(n_samples, self.n_features),
        )
        X.sum_duplicates()  # also sorts the indices

        return X

    def _more_tags(self):
        return {"X_types": [self.input_type]}


class BloomFilterFeatureHashers(TransformerMixin, BaseEstimator):
    """Implements feature hashing, aka the hashing trick.
    This class turns sequences of symbolic feature names (strings) into
    scipy.sparse matrices, using a hash function to compute the matrix column
    corresponding to a name. The hash function employed is the signed 32-bit
    version of Murmurhash3.
    Feature names of type byte string are used as-is. Unicode strings are
    converted to UTF-8 first, but no Unicode normalization is done.
    Feature values must be (finite) numbers.
    This class is a low-memory alternative to DictVectorizer and
    CountVectorizer, intended for large-scale (online) learning and situations
    where memory is tight, e.g. when running prediction code on embedded
    devices.
    Read more in the :ref:`User Guide <feature_hashing>`.
    .. versionadded:: 0.13
    Parameters
    ----------
    n_features : int, default=2**20
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    input_type : str, default='dict'
        Choose a string from {'dict', 'pair', 'string'}.
        Either "dict" (the default) to accept dictionaries over
        (feature_name, value); "pair" to accept pairs of (feature_name, value);
        or "string" to accept single strings.
        feature_name should be a string, while value should be a number.
        In the case of "string", a value of 1 is implied.
        The feature_name is hashed to find the appropriate column for the
        feature. The value's sign might be flipped in the output (but see
        non_negative, below).
    dtype : numpy dtype, default=np.float64
        The type of feature values. Passed to scipy.sparse matrix constructors
        as the dtype argument. Do not set this to bool, np.boolean or any
        unsigned integer type.
    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.
        .. versionchanged:: 0.19
            ``alternate_sign`` replaces the now deprecated ``non_negative``
            parameter.
    See Also
    --------
    DictVectorizer : Vectorizes string-valued features using a hash table.
    sklearn.preprocessing.OneHotEncoder : Handles nominal/categorical features.
    Examples
    --------
    >>> from sklearn.feature_extraction import FeatureHasher
    >>> h = FeatureHasher(n_features=10)
    >>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    >>> f = h.transform(D)
    >>> f.toarray()
    array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
           [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])
    """

    _parameter_constraints: dict = {
        "n_features": [Interval(Integral, 1, np.iinfo(np.int32).max, closed="both")],
        "input_type": [StrOptions({"dict", "pair", "string"})],
        "dtype": "no_validation",  # delegate to numpy
        "alternate_sign": ["boolean"],
    }

    def __init__(
        self,
        n_features=(2**20),
        *,
        input_type="dict",
        dtype=np.float64,
        alternate_sign=True,
        bloom_count=1,
        bloom_filter_error_rate=0.05,
        bloom_strat_type="chi",
        min_term_count=None,
    ):
        self.dtype = dtype
        self.input_type = input_type
        self.n_features = n_features
        self.alternate_sign = alternate_sign

        self.min_term_count = min_term_count
        self.bloom_count = bloom_count
        self.bloom_filter_error_rate = bloom_filter_error_rate
        self.bloom_strat_type = bloom_strat_type

    def fit(self, X=None, y=None):
        """No-op.

        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        # repeat input validation for grid search (which calls set_params)
        self._validate_params(self.n_features, self.input_type)

        X = iter(X)
        if self.input_type == "dict":
            X = (_iteritems(d) for d in X)
        elif self.input_type == "string":
            X = (((f, 1) for f in x) for x in X)

        # Count the vocabulary
        vocab = {}
        y_mean = 0
        for x, Y in zip(X, y):
            y_mean += Y
            for v, c in x:
                # Add the features to the vocabulary
                if v is not None:
                    if v not in vocab:
                        vocab[v] = {"c": c, "t_sum": Y * c}
                    else:
                        vocab[v]["c"] += c
                        vocab[v]["c_pos"] += Y * c
        y_mean = y_mean / len(y)

        if self.min_term_count and self.min_term_count > 1:
            # Apply minimum count filter
            feature_names = vocab.keys()
            for feature_name in feature_names:
                if vocab[feature_name]["c"] < self.min_term_count:
                    del vocab[feature_name]

        if self.bloom_count > 1:
            # Add feature weights for stratification
            if self.bloom_strat_type == "chi":
                # Use Chi (observed - expected) to find the feature weight
                # for stratification
                observed = (x["c_pos"] for x in vocab.values())
                expected = (x["c"] for x in vocab.values()) * y_mean
                rank = observed - expected

            elif self.bloom_strat_type == "lr":
                # TODO: OR use Logistic Regression to learn the feature weight
                raise NotImplementedError("Logistic Regression is not implemented yet.")

            else:
                raise ValueError("Unknown bloom_strat_type: %s" % self.bloom_strat_type)

            # Order the feature_names array by the feature weight
            feature_names = list(vocab.keys())
            feature_ranks = list(rank)
            feature_ranks, feature_names = zip(
                *sorted(zip(feature_ranks, feature_names))
            )
        else:
            # No stratification
            feature_names = list(vocab.keys())
            feature_ranks = [0] * len(feature_names)

        # Debug logging. TODO: Remove.
        self._feature_names = feature_names
        self._feature_ranks = feature_ranks

        # Now build the bloom filter vocabularies
        self.bloom_filters = []
        self.hash_bags = []
        bucket_size = int(len(feature_names) / self.bloom_count)

        for i in range(self.bloom_count):
            # Build the list of bloom filters, matching to each set of features
            # in the bucket
            if i == self.bloom_count - 1:
                # Last bucket may have more features
                f = feature_names[i * bucket_size :]
            else:
                f = feature_names[i * bucket_size : (i + 1) * bucket_size]

            # Create and fit the bloom filter
            bloom = BloomFilter(
                max_elements=len(f), error_rate=self.bloom_filter_error_rate
            )
            for feature_name in f:
                bloom.add(feature_name)
            self.bloom_filters.append(bloom)

            # Create and fit the hash bag
            hash_bag = FeatureHasher(
                n_features=self.n_features,
                alternate_sign=self.alternate_sign,
                input_type="string",
            )
            self.hash_bags.append(hash_bag)

        return self

    def transform(self, X):
        """Transform a sequence of instances to a scipy.sparse matrix.
        Parameters
        ----------
        X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.
        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        # Process everything as sparse regardless of setting
        X = iter(X)
        if self.input_type == "dict":
            X = (_iteritems(d) for d in X)
        elif self.input_type == "string":
            X = (((f, 1) for f in x) for x in X)

        X_by_hashbag = [None] * len(self.bloom_filters)

        for row, x in enumerate(X):
            x = list(x)

            # Split the features into the appropriate buckets by the bloom filters
            for i, bloom in enumerate(self.bloom_filters):
                if not X_by_hashbag[i]:
                    X_by_hashbag[i] = []
                X_by_hashbag[i].append([])
                for v, c in x:
                    if v in bloom:
                        X_by_hashbag[i][-1] += [v] * c

        # Hash the features
        X_by_hashbag = [
            self.hash_bags[i].transform(X_by_hashbag[i])
            for i in range(len(self.bloom_filters))
        ]

        # Merge the array of sparse matrices into a single sparse matrix
        X = sp.hstack(X_by_hashbag)

        return X

    def _more_tags(self):
        return {"X_types": [self.input_type]}
