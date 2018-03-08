import numpy as np
from .test_pfa_sklearn import _classification_task
from sklearn_to_pfa.mixed_nb import MixedNB
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def test_mixednb_all_nominals():
    """Check that MixedNB is equivalent to MultinomialNB for nominal variables."""
    X, y, types = _classification_task()

    # convert some features to nominal ones (one-hot encoded)
    X = (X > 0).astype(int)

    mixed_nb = MixedNB(is_nominal=[True] * 5)
    mixed_nb.partial_fit(X, y, classes=['a', 'b', 'c'])
    mixed_pred = mixed_nb.predict_proba(X)

    multi_nb = MultinomialNB()
    multi_nb.partial_fit(X, y, classes=['a', 'b', 'c'])
    multi_pred = multi_nb.predict_proba(X)

    assert np.allclose(mixed_pred, multi_pred)


def test_mixednb_all_continuous():
    """Check that MixedNB is equivalent to GaussNB for continuous variables."""
    X, y, types = _classification_task()

    mixed_nb = MixedNB(is_nominal=[False] * 5)
    mixed_nb.partial_fit(X, y, classes=['a', 'b', 'c'])
    mixed_pred = mixed_nb.predict_proba(X)

    gauss_nb = GaussianNB()
    gauss_nb.partial_fit(X, y, classes=['a', 'b', 'c'])
    gauss_pred = gauss_nb.predict_proba(X)

    assert np.allclose(mixed_pred, gauss_pred)
