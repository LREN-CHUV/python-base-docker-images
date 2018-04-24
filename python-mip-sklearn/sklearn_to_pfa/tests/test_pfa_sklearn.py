# TODO: run pytest command `python -m pytest tests/test_pfa_sklearn.py --capture=no` from docker

import pytest
import numpy as np
import pandas as pd
from titus.genpy import PFAEngine
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn_to_pfa.sklearn_to_pfa import sklearn_to_pfa
from sklearn_to_pfa.mixed_nb import MixedNB


def _sgd_regressor(X, y):
    estimator = SGDRegressor()
    estimator.partial_fit(X, y)
    return estimator


def _mlp_regressor(X, y):
    estimator = MLPRegressor(hidden_layer_sizes=(3, 3))
    estimator.partial_fit(X, y)
    return estimator


def _sgd_classifier(X, y, **kwargs):
    estimator = SGDClassifier()
    estimator.partial_fit(X, y, **kwargs)
    return estimator


def _mlp_classifier(X, y, **kwargs):
    estimator = MLPClassifier(hidden_layer_sizes=(3, 3))
    estimator.partial_fit(X, y, **kwargs)
    return estimator


def _multinomialnb(X, y, **kwargs):
    estimator = MultinomialNB()
    estimator.partial_fit(X, y, **kwargs)
    return estimator


def _gaussiannb(X, y, **kwargs):
    estimator = GaussianNB()
    estimator.partial_fit(X, y, **kwargs)
    return estimator


def _mixednb(X, y, is_nominal, **kwargs):
    estimator = MixedNB(is_nominal=is_nominal)
    estimator.partial_fit(X, y, **kwargs)
    return estimator


def _kmeans(X, **kwargs):
    estimator = KMeans(**kwargs)
    estimator.fit(X)
    return estimator


def _kneighborsregressor(X, y, **kwargs):
    estimator = KNeighborsRegressor(**kwargs)
    estimator.fit(X, y)
    return estimator


def _kneighborsclassifier(X, y, **kwargs):
    estimator = KNeighborsClassifier(**kwargs)
    estimator.fit(X, y)
    return estimator


def _gradientboostingregressor(X, y, **kwargs):
    estimator = GradientBoostingRegressor(**kwargs)
    estimator.fit(X, y)
    return estimator


def _gradientboostingclassifier(X, y, **kwargs):
    estimator = GradientBoostingClassifier(**kwargs)
    estimator.fit(X, y)
    return estimator


def _predict_pfa(X, types, pfa):
    engine, = PFAEngine.fromJson(pfa)
    columns = [c for c, _ in types]

    pfa_pred = []
    for x in X:
        p = {}
        for col, e, (_, typ) in zip(columns, x, types):
            if typ == 'integer':
                p[col] = int(e)
            else:
                p[col] = e

        pfa_pred.append(engine.action(p))
    return np.array(pfa_pred)


def _regression_task(n_features=10):
    X, y = datasets.make_regression(n_samples=100, n_features=n_features)
    types = [('feature{}'.format(i), 'double') for i in range(n_features)]
    return X, y, types


def _arrays_equal(x, y):
    return all(abs(x - y) < 1e-5)


def test_estimator_to_pfa_sgd_regressor():
    """Check that converted PFA is giving the same results as SGDRegressor"""
    X, y, types = _regression_task()
    estimator = _sgd_regressor(X, y)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert _arrays_equal(estimator_pred, pfa_pred)


def test_estimator_to_pfa_mlp_regressor():
    """Check that converted PFA is giving the same results as MLPRegressor"""
    X, y, types = _regression_task()
    estimator = _mlp_regressor(X, y)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert _arrays_equal(estimator_pred, pfa_pred)


def _classification_task(n_features=5, dtypes=None):
    X, y = datasets.make_classification(n_samples=100, n_features=n_features, n_redundant=0, n_informative=n_features, n_classes=3)
    y = pd.Series(y).map({0: 'a', 1: 'b', 2: 'c'}).values
    types = [('feature{}'.format(i), 'double') for i in range(n_features)]

    if dtypes is not None:
        for i, t in enumerate(dtypes):
            # integer
            if t == 'i':
                X[:, i] = X[:, i].astype(int)
                types[i] = ('feature{}'.format(i), 'integer')
            # nominal
            elif t == 'n':
                # convert some features to nominal ones (one-hot encoded)
                X[:, i] = (X[:, i] > 0).astype(int)
                types[i] = ('feature{}'.format(i), 'integer')
            elif t == 'c':
                pass
            else:
                raise NotImplementedError()

    return X, y, types


def test_estimator_to_pfa_sgd_classifier():
    """Check that converted PFA is giving the same results as SGDClassifier"""
    X, y, types = _classification_task()
    estimator = _sgd_classifier(X, y, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_mlp_classifier():
    """Check that converted PFA is giving the same results as MLPClassifier"""
    X, y, types = _classification_task()
    estimator = _mlp_classifier(X, y, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_multinomialnb():
    """Check that converted PFA is giving the same results as MultinomialNB"""
    X, y, types = _classification_task()

    # artifically create 0, 1 inputs from X because `MultinomialNB` works only with counts
    X = (X > 0).astype(int)

    estimator = _multinomialnb(X, y, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_gaussiannb():
    """Check that converted PFA is giving the same results as GaussianNB"""
    X, y, types = _classification_task()

    estimator = _gaussiannb(X, y, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


@pytest.mark.parametrize('dtypes', ['nnnnn', 'cccnn', 'cccin'])
def test_estimator_to_pfa_mixednb(dtypes):
    """Check that converted PFA is giving the same results as MixedNB"""
    X, y, types = _classification_task(dtypes=dtypes)

    is_nominal = [t == 'n' for t in dtypes]
    estimator = _mixednb(X, y, is_nominal=is_nominal, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_mixednb_zero_prior():
    """Check that converted PFA is giving the same results as MultinomialNB with category that has no values."""
    dtypes = 'ccn'
    X, y, types = _classification_task(n_features=3, dtypes=dtypes)
    y[:] = 'a'

    is_nominal = [t == 'n' for t in dtypes]
    estimator = _mixednb(X, y, is_nominal=is_nominal, classes=['a', 'b', 'c'])

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_kmeans():
    """Check that converted PFA is giving the same results as KMeans"""
    X, _, types = _classification_task()

    estimator = _kmeans(X, n_clusters=2)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_kneighborsregressor():
    """Check that converted PFA is giving the same results as KNeighborsRegressor"""
    X, y, types = _regression_task()

    estimator = _kneighborsregressor(X, y, n_neighbors=2)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_kneighborsclassifier():
    """Check that converted PFA is giving the same results as KNeighborsClassifier"""
    X, y, types = _classification_task()

    estimator = _kneighborsclassifier(X, y, n_neighbors=2)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_gradientboostingregressor():
    """Check that converted PFA is giving the same results as GradientBoostingRegressor"""
    X, y, types = _regression_task()

    estimator = _gradientboostingregressor(X, y, n_estimators=10, learning_rate=0.1)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    np.testing.assert_almost_equal(estimator_pred, pfa_pred, decimal=5)


def test_estimator_to_pfa_gradientboostingclassifier():
    """Check that converted PFA is giving the same results as GradientBoostingClassifier"""
    X, y, types = _classification_task()

    estimator = _gradientboostingclassifier(X, y, n_estimators=10, learning_rate=0.1)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)


def test_estimator_to_pfa_gradientboostingclassifier_nosplits():
    X, y, types = _classification_task()

    # `min_samples_split` guarantees there will be no splits
    estimator = _gradientboostingclassifier(X, y, min_samples_split=1000000, n_estimators=10, learning_rate=0.1)

    pfa = sklearn_to_pfa(estimator, types)

    estimator_pred = estimator.predict(X)
    pfa_pred = _predict_pfa(X, types, pfa)

    assert all(estimator_pred == pfa_pred)
