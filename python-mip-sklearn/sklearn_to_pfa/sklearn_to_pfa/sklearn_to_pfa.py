# Copyright (C) 2017  LREN CHUV for Human Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sklearn
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from .mixed_nb import MixedNB
import os
import logging
import numpy as np
import titus.prettypfa
import jinja2


def sklearn_to_pfa(estimator, types, featurizer=None):
    """
    Convert scikit-learn estimator to PFA format.
    :param estimator: Scikit-learn estimator, must be supported
    :param types: List of tuples (name, type)
    """
    types = _fix_types_compatibility(types)

    featurizer = featurizer or _construct_featurizer(types)

    if isinstance(estimator, SGDRegressor):
        return _pfa_sgdregressor(estimator, types, featurizer)
    elif isinstance(estimator, SGDClassifier):
        return _pfa_sgdclassifier(estimator, types, featurizer)
    elif isinstance(estimator, MLPRegressor):
        return _pfa_mlpregressor(estimator, types, featurizer)
    elif isinstance(estimator, MLPClassifier):
        return _pfa_mlpclassifier(estimator, types, featurizer)
    elif isinstance(estimator, MultinomialNB):
        return _pfa_multinomialnb(estimator, types, featurizer)
    elif isinstance(estimator, GaussianNB):
        return _pfa_gaussiannb(estimator, types, featurizer)
    elif isinstance(estimator, MixedNB):
        return _pfa_mixednb(estimator, types, featurizer)
    elif isinstance(estimator, KMeans):
        return _pfa_kmeans(estimator, types, featurizer)
    elif isinstance(estimator, KNeighborsRegressor):
        return _pfa_kneighborsregressor(estimator, types, featurizer)
    elif isinstance(estimator, KNeighborsClassifier):
        return _pfa_kneighborsclassifier(estimator, types, featurizer)
    elif isinstance(estimator, GradientBoostingRegressor):
        return _pfa_gradientboostingregressor(estimator, types, featurizer)
    elif isinstance(estimator, GradientBoostingClassifier):
        return _pfa_gradientboostingclassifier(estimator, types, featurizer)
    else:
        raise NotImplementedError('Estimator {} is not yet supported'.format(estimator.__class__.__name__))


def _pfa_sgdregressor(estimator, types, featurizer):
    # construct template
    pretty_pfa = """{% extends "regressor.ppfa" %}
{% block types %}
Regression = record(Regression, const: double, coeff: array(double));
{% endblock %}

{% block cells %}
model(Regression) = {const: 0.0, coeff: []};
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};
model.reg.linear(x, model)
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['model']['init'] = {'const': estimator.intercept_[0], 'coeff': list(estimator.coef_)}

    return pfa


def _pfa_sgdclassifier(estimator, types, featurizer):
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
Regression = record(Regression, const: double, coeff: array(double));
{% endblock %}

{% block cells %}
model(array(Regression)) = [];
classes(array(string)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};
var scores = a.map(model, fcn(r: Regression -> double) model.reg.linear(x, r));
classes[a.argmax(scores)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    pfa['cells']['model']['init'] = [
        {
            'const': const,
            'coeff': list(coeff)
        } for const, coeff in zip(estimator.intercept_, estimator.coef_)
    ]

    return pfa


def _pfa_mlpregressor(estimator, types, featurizer):
    """
    See https://github.com/opendatagroup/hadrian/wiki/Basic-neural-network
    """
    # construct template
    pretty_pfa = """{% extends "regressor.ppfa" %}
{% block types %}
Layer = record(Layer,
               weights: array(array(double)),
               bias: array(double));
{% endblock %}

{% block cells %}
neuralnet(array(Layer)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};
var activation = model.neural.simpleLayers(x, neuralnet, fcn(x: double -> double) m.link.relu(x));
activation[0]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    # NOTE: `model.neural.simpleLayers` accepts transposed matrices
    pfa['cells']['neuralnet']['init'] = [
        {
            'bias': bias.tolist(),
            'weights': weights.T.tolist()
        } for bias, weights in zip(estimator.intercepts_, estimator.coefs_)
    ]

    return pfa


def _pfa_mlpclassifier(estimator, types, featurizer):
    """
    See https://github.com/opendatagroup/hadrian/wiki/Basic-neural-network
    """
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
Layer = record(Layer,
               weights: array(array(double)),
               bias: array(double));
{% endblock %}

{% block cells %}
neuralnet(array(Layer)) = [];
classes(array(string)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};
var activations = model.neural.simpleLayers(x, neuralnet, fcn(x: double -> double) m.link.relu(x));
classes[a.argmax(activations)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    # NOTE: `model.neural.simpleLayers` accepts transposed matrices
    pfa['cells']['neuralnet']['init'] = [
        {
            'bias': bias.tolist(),
            'weights': weights.T.tolist()
        } for bias, weights in zip(estimator.intercepts_, estimator.coefs_)
    ]

    return pfa


def _pfa_multinomialnb(estimator, types, featurizer):
    """
    See https://github.com/opendatagroup/hadrian/wiki/Basic-naive-bayes
    NOTE: in our use case we use mostly one-hot encoded variables, so using BernoulliNB might make
        more sense
    """
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
Distribution = record(Distribution,
                    logLikelihoods: array(double),
                    logPrior: double);
{% endblock %}

{% block cells %}
model(array(Distribution)) = [];
classes(array(string)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var classLL = a.map(model, fcn(dist: Distribution -> double) {
    model.naive.multinomial(x, dist.logLikelihoods) + dist.logPrior
});
var norm = a.logsumexp(classLL);
var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
classes[a.argmax(probs)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    logprior = np.maximum(estimator.class_log_prior_, -1e10)
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    pfa['cells']['model']['init'] = [
        {
            'logLikelihoods': ll.tolist(),
            'logPrior': log_prior.tolist()
        } for log_prior, ll in zip(logprior, np.exp(estimator.feature_log_prob_))
    ]

    return pfa


def _pfa_gaussiannb(estimator, types, featurizer):
    """
    See https://github.com/opendatagroup/hadrian/wiki/Basic-naive-bayes
    """
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
Distribution = record(Distribution,
                      stats: array(record(M, mean: double, variance: double)),
                      logPrior: double);
{% endblock %}

{% block cells %}
model(array(Distribution)) = [];
classes(array(string)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var classLL = a.map(model, fcn(dist: Distribution -> double) {
  model.naive.gaussian(x, dist.stats) + dist.logPrior
});

var norm = a.logsumexp(classLL);
var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
classes[a.argmax(probs)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    prior = np.maximum(estimator.class_prior_, 1e-10)
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    pfa['cells']['model']['init'] = [
        {
            'stats': [{
                'mean': m,
                'variance': s
            } for m, s in zip(means, sigmas)],
            'logPrior': np.log(prior).tolist()
        } for prior, means, sigmas in zip(prior, estimator.theta_, estimator.sigma_)
    ]

    return pfa


def _pfa_mixednb(estimator, types, featurizer):
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
GaussianDistribution = record(GaussianDistribution,
                              stats: array(record(M, mean: double, variance: double)));
MultinomialDistribution = record(MultinomialDistribution,
                                 logLikelihoods: array(double));
{% endblock %}

{% block cells %}
gaussModel(array(GaussianDistribution)) = [];
multinomialModel(array(MultinomialDistribution)) = [];
classes(array(string)) = [];
logPrior(array(double)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var gaussFeatures = if( a.len(gaussModel) > 0 ) a.len(gaussModel[0,"stats"]) else 0;

var gaussianLL = a.map(gaussModel, fcn(dist: GaussianDistribution -> double) {
    model.naive.gaussian(a.subseq(x, 0, gaussFeatures), dist.stats)
});

var multinomialLL = a.map(multinomialModel, fcn(dist: MultinomialDistribution -> double) {
    model.naive.multinomial(a.subseq(x, gaussFeatures, a.len(x)), dist.logLikelihoods)
});

var classLL = logPrior;
if (a.len(gaussianLL) > 0)
    classLL = la.add(classLL, gaussianLL);
if (a.len(multinomialLL) > 0)
    classLL = la.add(classLL, multinomialLL);

var norm = a.logsumexp(classLL);
var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
classes[a.argmax(probs)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    # avoid null values from log prior
    logprior = np.maximum(estimator.class_log_prior_, -1e10)
    pfa['cells']['logPrior']['init'] = logprior.tolist()

    # assumes that continuous features go before nominal ones
    if hasattr(estimator.gauss_nb, 'theta_'):
        pfa['cells']['gaussModel']['init'] = [
            {
                'stats': [{
                    'mean': m,
                    'variance': s
                } for m, s in zip(means, sigmas)]
            } for means, sigmas in zip(estimator.gauss_nb.theta_, estimator.gauss_nb.sigma_)
        ]
    if hasattr(estimator.multi_nb, 'feature_log_prob_'):
        pfa['cells']['multinomialModel']['init'] = [
            {
                'logLikelihoods': ll.tolist()
            } for ll in np.exp(estimator.multi_nb.feature_log_prob_)
        ]

    return pfa


def _pfa_kmeans(estimator, types, featurizer):
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block output %}
output: int
{% endblock %}

{% block types %}
Cluster = record(Cluster, center: array(double), id: int);
{% endblock %}

{% block cells %}
clusters(array(Cluster)) = [];
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var cluster = model.cluster.closest(x, clusters, metric.simpleEuclidean);
cluster.id
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['clusters']['init'] = [
        {
            'center': c.tolist(),
            'id': i
        } for i, c in enumerate(estimator.cluster_centers_)
    ]
    return pfa


def _pfa_kneighborsregressor(estimator, types, featurizer):
    """See https://github.com/opendatagroup/hadrian/wiki/Basic-nearest-neighbors"""
    # construct template
    pretty_pfa = """{% extends "regressor.ppfa" %}
{% block types %}
Point = record(Point,
               x: array(double),
               y: double);
Codebook = array(Point);
{% endblock %}

{% block cells %}
codebook(Codebook) = [];
nNeighbors(int) = 5;
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var neighbors = model.neighbor.nearestK(nNeighbors, x, codebook, fcn(x: array(double), p: Point -> double) {
    metric.simpleEuclidean(x, p.x)
});
a.mean(a.map(neighbors, fcn(p: Point -> double) p.y))
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['codebook']['init'] = [
        {
            'x': x.tolist(),
            'y': y
        } for x, y in zip(estimator._fit_X, estimator._y)
    ]
    pfa['cells']['nNeighbors']['init'] = estimator.n_neighbors

    return pfa


def _pfa_kneighborsclassifier(estimator, types, featurizer):
    """See https://github.com/opendatagroup/hadrian/wiki/Basic-nearest-neighbors"""
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
Point = record(Point,
               x: array(double),
               y: string);
Codebook = array(Point);
{% endblock %}

{% block cells %}
codebook(Codebook) = [];
nNeighbors(int) = 5;
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var neighbors = model.neighbor.nearestK(nNeighbors, x, codebook, fcn(x: array(double), p: Point -> double) {
    metric.simpleEuclidean(x, p.x)
});
a.mode(a.map(neighbors, fcn(p: Point -> string) p.y))
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    y_labels = [estimator.classes_[e] for e in estimator._y]
    pfa['cells']['codebook']['init'] = [
        {
            'x': x.tolist(),
            'y': y
        } for x, y in zip(estimator._fit_X, y_labels)
    ]
    pfa['cells']['nNeighbors']['init'] = estimator.n_neighbors

    return pfa


def make_tree(tree, node_id=0):
    if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
        leaf_value = float(tree.value[node_id][0, 0])

        # special case for empty tree with just root
        if node_id == 0:
            return {'TreeNode': {
                'feature': 0,
                'operator': '<=',
                'value': 0.,
                'pass': {'double': leaf_value},
                'fail': {'double': leaf_value}
            }}
        else:
            return {'double': leaf_value}

    return {'TreeNode': {
        'feature': int(tree.feature[node_id]),
        'operator': '<=',
        'value': float(tree.threshold[node_id]),
        'pass': make_tree(tree, tree.children_left[node_id]),
        'fail': make_tree(tree, tree.children_right[node_id])
    }}


def _pfa_gradientboostingregressor(estimator, types, featurizer):
    """See https://github.com/opendatagroup/hadrian/wiki/Basic-decision-tree"""
    # construct template
    pretty_pfa = """{% extends "regressor.ppfa" %}
{% block types %}
TreeNode = record(TreeNode,
                feature: int,
                operator: string,
                value: double,
                pass: union(double, TreeNode),
                fail: union(double, TreeNode));
Row = record(Row, values: array(double));
{% endblock %}

{% block cells %}
// empty tree to satisfy type constraint; will be filled in later
trees(array(TreeNode)) = [];
// model intercept to which tree predictions are added
intercept(double) = 0.0;
learningRate(double) = 0.0;
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var row = new(Row, values: x);

var scores = a.map(trees, fcn(tree: TreeNode -> double) {
  model.tree.simpleWalk(row, tree, fcn(d: Row, t: TreeNode -> boolean) {
    d.values[t.feature] <= t.value
  })
});

intercept + learningRate * a.sum(scores)
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    tree_dicts = []
    for tree in estimator.estimators_[:, 0]:
        tree_dicts.append(make_tree(tree.tree_)['TreeNode'])
    pfa["cells"]["trees"]["init"] = tree_dicts
    pfa['cells']['intercept']['init'] = estimator.init_.mean
    pfa['cells']['learningRate']['init'] = estimator.learning_rate

    return pfa


def _pfa_gradientboostingclassifier(estimator, types, featurizer):
    """See https://github.com/opendatagroup/hadrian/wiki/Basic-decision-tree"""
    # construct template
    pretty_pfa = """{% extends "classifier.ppfa" %}
{% block types %}
TreeNode = record(TreeNode,
                feature: int,
                operator: string,
                value: double,
                pass: union(double, TreeNode),
                fail: union(double, TreeNode));
Row = record(Row, values: array(double));
{% endblock %}

{% block cells %}
classes(array(string)) = [];
// set of trees for each class
classesTrees(array(array(TreeNode))) = [];
// model priors to which tree predictions are added
priors(array(double)) = [];
learningRate(double) = 0.0;
{% endblock %}

{% block action %}
var x = {{featurizer | safe}};

var row = new(Row, values: x);

// trees activations
var activations = a.map(classesTrees, fcn(trees: array(TreeNode) -> double) {
    var scores = a.map(trees, fcn(tree: TreeNode -> double) {
      model.tree.simpleWalk(row, tree, fcn(d: Row, t: TreeNode -> boolean) {
        d.values[t.feature] <= t.value
      })
    });
    learningRate * a.sum(scores)
});

// add priors
activations = la.add(priors, activations);

// probabilities
var norm = a.logsumexp(activations);
var probs = a.map(activations, fcn(x: double -> double) m.exp(x - norm));

classes[a.argmax(probs)]
{% endblock %}
    """

    pretty_pfa = _render(pretty_pfa, types, featurizer=featurizer)

    # compile
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    tree_dicts = [[make_tree(tree.tree_)['TreeNode'] for tree in trees] for trees in estimator.estimators_.T]
    pfa["cells"]["classesTrees"]["init"] = tree_dicts
    pfa['cells']['classes']['init'] = list(estimator.classes_)
    pfa['cells']['priors']['init'] = list(estimator.init_.priors)
    pfa['cells']['learningRate']['init'] = estimator.learning_rate

    return pfa


def _construct_featurizer(types):
    inputs = []
    for name, typ in types:
        if typ == 'int':
            inputs.append('cast.double(input.{})'.format(name))
        else:
            inputs.append('input.{}'.format(name))

    return """
new(array(double),
    {inputs}
    )
    """.format(inputs=',\n'.join(inputs)).strip()


def _fix_types_compatibility(types):
    new_types = []
    for name, typ in types:
        if typ == 'real':
            typ = 'double'
        elif typ == 'integer':
            typ = 'int'
        elif typ in ('polynominal', 'binominal'):
            typ = 'string'
        new_types.append((name, typ))
    return new_types


def _render(tpl, types, **context):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(dir_path, 'templates')), autoescape=True
    ).from_string(tpl).render(types=types, **context)
