# Author: Lars Buitinck
# License: BSD 3 clause

import numbers
from numbers import Number
from bloom_filter2 import BloomFilter

from collections.abc import Mapping, Iterable

import numpy as np
import scipy.sparse as sp

from ..utils import IS_PYPY
from ..base import BaseEstimator, TransformerMixin

if not IS_PYPY:
    from ._hashing_fast import transform as _hashing_transform
else:

    def _hashing_transform(*args, **kwargs):
        raise NotImplementedError(
            "FeatureHasher is not compatible with PyPy (see "
            "https://github.com/scikit-learn/scikit-learn/issues/11540 "
            "for the status updates)."
        )


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

    def __init__(
        self,
        n_features=(2 ** 20),
        *,
        input_type="dict",
        dtype=np.float64,
        alternate_sign=True,
    ):
        self._validate_params(n_features, input_type)

        self.dtype = dtype
        self.input_type = input_type
        self.n_features = n_features
        self.alternate_sign = alternate_sign

    @staticmethod
    def _validate_params(n_features, input_type):
        # strangely, np.int16 instances are not instances of Integral,
        # while np.int64 instances are...
        if not isinstance(n_features, numbers.Integral):
            raise TypeError(
                "n_features must be integral, got %r (%s)."
                % (n_features, type(n_features))
            )
        elif n_features < 1 or n_features >= np.iinfo(np.int32).max + 1:
            raise ValueError("Invalid number of features (%d)." % n_features)

        if input_type not in ("dict", "pair", "string"):
            raise ValueError(
                "input_type must be 'dict', 'pair' or 'string', got %r." % input_type
            )

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
            raw_X = (((f, 1) for f in x) for x in raw_X)
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

    def __init__(
        self,
        n_features=(2 ** 20),
        *,
        input_type="dict",
        dtype=np.float64,
        alternate_sign=True,
        bloom_count=1,
        bloom_filter_error_rate=0.05,
        min_term_count=None,
    ):
        self._validate_params(n_features, input_type)

        self.dtype = dtype
        self.input_type = input_type
        self.n_features = n_features
        self.alternate_sign = alternate_sign

        self.min_term_count = min_term_count
        self.bloom_count = bloom_count
        self.bloom_filter_error_rate = bloom_filter_error_rate
        

    @staticmethod
    def _validate_params(n_features, input_type):
        # strangely, np.int16 instances are not instances of Integral,
        # while np.int64 instances are...
        if not isinstance(n_features, numbers.Integral):
            raise TypeError(
                "n_features must be integral, got %r (%s)."
                % (n_features, type(n_features))
            )
        elif n_features < 1 or n_features >= np.iinfo(np.int32).max + 1:
            raise ValueError("Invalid number of features (%d)." % n_features)

        if input_type not in ("dict", "pair", "string"):
            raise ValueError(
                "input_type must be 'dict', 'pair' or 'string', got %r." % input_type
            )

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
        for x, Y in zip(X, y):
            for v, c in x:
                # Add the features to the vocabulary
                if v is not None:
                    if v not in vocab:
                        vocab[v] = {"c":c , "t_sum":Y * c}
                    else:
                        vocab[v]["c"] += c
                        vocab[v]["t_sum"] += Y * c

        if self.min_term_count and self.min_term_count > 1:
            # Apply minimum count filter
            keys = vocab.keys()
            for feature_name in feature_names:
                if vocab[feature_name]["c"] < self.min_term_count:
                    del vocab[feature_name]
        
        # Extract the keys and target sum
        feature_names = list(vocab.keys())
        feature_target_sums = [x["t_sum"] for x in vocab.values()]

        # Build the vocabulary per bloom filter bucket
        if self.bloom_count > 1:
            # Sorting is needed by total target weight per feature
            feature_target_sums, feature_names = zip(*sorted(zip(feature_target_sums, feature_names)))

        # Debug logging. TODO: Remove.
        self._feature_names = feature_names
        self._feature_target_sums = feature_target_sums
        
        # Now build the bloom filter vocabularies
        self.bloom_filters = []
        self.hash_bags = []
        bucket_size = int(len(feature_names) / self.bloom_count)

        for i in range(self.bloom_count):
            # Build the list of bloom filters, matching to each set of features
            # in the bucket
            if i == self.bloom_count - 1:
                # Last bucket may have more features
                f = feature_names[i * bucket_size : ]
            else:
                f = feature_names[i * bucket_size : (i + 1) * bucket_size]

            # Create and fit the bloom filter
            bloom = BloomFilter(max_elements=len(f), error_rate=self.bloom_filter_error_rate)
            for feature_name in f:
                bloom.add(feature_name)
            self.bloom_filters.append(bloom)

            # Create and fit the hash bag
            hash_bag = FeatureHasher(n_features=self.n_features,
                alternate_sign=self.alternate_sign, input_type="string"
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
        X_by_hashbag = [self.hash_bags[i].transform(X_by_hashbag[i]) for i in range(len(self.bloom_filters))]

        # Merge the array of sparse matrices into a single sparse matrix
        X = sp.hstack(X_by_hashbag)

        return X

    def _more_tags(self):
        return {"X_types": [self.input_type]}
