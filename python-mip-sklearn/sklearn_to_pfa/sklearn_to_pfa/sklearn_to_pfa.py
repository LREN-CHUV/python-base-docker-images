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
import logging
import numpy as np
import titus.prettypfa


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
    """
    Example output:
        input: record(Data,
            feature0: double,
            feature1: double,
        )
        output: double
        action:
            -2.525797382870301 + 31.7004451488 * input.feature0 + 42.5005713274 * input.feature1
    """
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Regression = record(Regression, const: double, coeff: array(double));
    Input = {input_record}
input: Input
output: double
cells:
    // query(Query) = {{}};
    model(Regression) = {{const: 0.0, coeff: []}};
fcns:
    {functions}
action:
    var x = {featurizer};
    model.reg.linear(x, model)
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

    # compile
    logging.info(pretty_pfa)
    pfa = titus.prettypfa.jsonNode(pretty_pfa)

    # add model from scikit-learn
    pfa['cells']['model']['init'] = {'const': estimator.intercept_[0], 'coeff': list(estimator.coef_)}

    return pfa


def _pfa_sgdclassifier(estimator, types, featurizer):
    """
    Example output:
        types:
            Query = record(Query,
                           sql: string,
                           variable: string,
                           covariables: array(string));
            Input = record(Input,
                a: double,
                b: double
            );
            Regression = record(Regression, const: double, coeff: array(double));
        input: Input
        output: string
        cells:
            query(Query) = {sql: SELECT, variable: dep, covariables: [a, b, c]};
            model(array(Regression)) = [
                {const: 3, coeff: [1, 2]},
                {const: 3, coeff: [1, 4]}
            ];
            classes(array(string)) = [a, b, c];
        fcns:
            featurize = fcn(x: Input -> array(double))
                new(array(double),
                    x.a,
                    x.b
                )
        action:
            // TODO: define this as a function when functions start working with titus python 3
            var x = new(array(double),
                input.a,
                input.b
            );
            var scores = a.map(model, fcn(r: Regression -> double) model.reg.linear(x, r));
            classes[a.argmax(scores)]
    """
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Regression = record(Regression, const: double, coeff: array(double));
    Input = {input_record}
input: Input
output: string
cells:
    // query(Query) = {{}};
    model(array(Regression)) = [];
    classes(array(string)) = [];
fcns:
    {functions}
action:
    var x = {featurizer};
    var scores = a.map(model, fcn(r: Regression -> double) model.reg.linear(x, r));
    classes[a.argmax(scores)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Layer = record(Layer,
                   weights: array(array(double)),
                   bias: array(double));
    Input = {input_record}
input: Input
output: double
cells:
    neuralnet(array(Layer)) = [];
fcns:
{functions}
action:
    var x = {featurizer};
    var activation = model.neural.simpleLayers(x, neuralnet, fcn(x: double -> double) m.link.relu(x));
    activation[0]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Layer = record(Layer,
                   weights: array(array(double)),
                   bias: array(double));
    Input = {input_record}
input: Input
output: string
cells:
    neuralnet(array(Layer)) = [];
    classes(array(string)) = [];
fcns:
{functions}
action:
    var x = {featurizer};
    var activations = model.neural.simpleLayers(x, neuralnet, fcn(x: double -> double) m.link.relu(x));
    classes[a.argmax(activations)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Distribution = record(Distribution,
                        logLikelihoods: array(double),
                        logPrior: double);
    Input = {input_record}
input: Input
output: string
cells:
    model(array(Distribution)) = [];
    classes(array(string)) = [];
fcns:
{functions}
action:
    var x = {featurizer};

    var classLL = a.map(model, fcn(dist: Distribution -> double) {{
        model.naive.multinomial(x, dist.logLikelihoods) + dist.logPrior
    }});
    var norm = a.logsumexp(classLL);
    var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
    classes[a.argmax(probs)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Distribution = record(Distribution,
                          stats: array(record(M, mean: double, variance: double)),
                          logPrior: double);
    Input = {input_record}
input: Input
output: string
cells:
    model(array(Distribution)) = [];
    classes(array(string)) = [];
fcns:
{functions}
action:
    var x = {featurizer};

    var classLL = a.map(model, fcn(dist: Distribution -> double) {{
      model.naive.gaussian(x, dist.stats) + dist.logPrior
    }});

    var norm = a.logsumexp(classLL);
    var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
    classes[a.argmax(probs)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    GaussianDistribution = record(GaussianDistribution,
                                  stats: array(record(M, mean: double, variance: double)));
    MultinomialDistribution = record(MultinomialDistribution,
                                     logLikelihoods: array(double));
    Input = {input_record}
input: Input
output: string
cells:
    gaussModel(array(GaussianDistribution)) = [];
    multinomialModel(array(MultinomialDistribution)) = [];
    classes(array(string)) = [];
    logPrior(array(double)) = [];
fcns:
{functions}
action:
    var x = {featurizer};

    var gaussFeatures = if( a.len(gaussModel) > 0 ) a.len(gaussModel[0,"stats"]) else 0;

    var gaussianLL = a.map(gaussModel, fcn(dist: GaussianDistribution -> double) {{
        model.naive.gaussian(a.subseq(x, 0, gaussFeatures), dist.stats)
    }});

    var multinomialLL = a.map(multinomialModel, fcn(dist: MultinomialDistribution -> double) {{
        model.naive.multinomial(a.subseq(x, gaussFeatures, a.len(x)), dist.logLikelihoods)
    }});

    var classLL = logPrior;
    if (a.len(gaussianLL) > 0)
        classLL = la.add(classLL, gaussianLL);
    if (a.len(multinomialLL) > 0)
        classLL = la.add(classLL, multinomialLL);

    var norm = a.logsumexp(classLL);
    var probs = a.map(classLL, fcn(x: double -> double) m.exp(x - norm));
    classes[a.argmax(probs)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Cluster = record(Cluster, center: array(double), id: int);
    Input = {input_record}
input: Input
output: int
cells:
    clusters(array(Cluster)) = [];
fcns:
{functions}
action:
    var x = {featurizer};
    var cluster = model.cluster.closest(x, clusters, metric.simpleEuclidean);
    cluster.id
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Point = record(Point,
                   x: array(double),
                   y: double);
    Codebook = array(Point);
    Input = {input_record}
input: Input
output: double
cells:
    codebook(Codebook) = [];
    nNeighbors(int) = 5;
fcns:
{functions}
action:
    var x = {featurizer};
    var neighbors = model.neighbor.nearestK(nNeighbors, x, codebook, fcn(x: array(double), p: Point -> double) {{
        metric.simpleEuclidean(x, p.x)
    }});
    a.mean(a.map(neighbors, fcn(p: Point -> double) p.y))
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    Point = record(Point,
                   x: array(double),
                   y: string);
    Codebook = array(Point);
    Input = {input_record}
input: Input
output: string
cells:
    codebook(Codebook) = [];
    nNeighbors(int) = 5;
fcns:
{functions}
action:
    var x = {featurizer};
    var neighbors = model.neighbor.nearestK(nNeighbors, x, codebook, fcn(x: array(double), p: Point -> double) {{
        metric.simpleEuclidean(x, p.x)
    }});
    a.mode(a.map(neighbors, fcn(p: Point -> string) p.y))
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    TreeNode = record(TreeNode,
                    feature: int,
                    operator: string,
                    value: double,
                    pass: union(double, TreeNode),
                    fail: union(double, TreeNode));
    Row = record(Row, values: array(double));
    Input = {input_record}
input: Input
output: double
cells:
    // empty tree to satisfy type constraint; will be filled in later
    trees(array(TreeNode)) = [];
    // model intercept to which tree predictions are added
    intercept(double) = 0.0;
    learningRate(double) = 0.0;
fcns:
{functions}
action:
    var x = {featurizer};
    var row = new(Row, values: x);

    var scores = a.map(trees, fcn(tree: TreeNode -> double) {{
      model.tree.simpleWalk(row, tree, fcn(d: Row, t: TreeNode -> boolean) {{
        d.values[t.feature] <= t.value
      }})
    }});

    intercept + learningRate * a.sum(scores)
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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
    input_record = _input_record(types)

    # construct template
    pretty_pfa = """
types:
    Query = record(Query,
                   sql: string,
                   variable: string,
                   covariables: array(string));
    TreeNode = record(TreeNode,
                    feature: int,
                    operator: string,
                    value: double,
                    pass: union(double, TreeNode),
                    fail: union(double, TreeNode));
    Row = record(Row, values: array(double));
    Input = {input_record}
input: Input
output: string
cells:
    classes(array(string)) = [];
    // set of trees for each class
    classesTrees(array(array(TreeNode))) = [];
    // model priors to which tree predictions are added
    priors(array(double)) = [];
    learningRate(double) = 0.0;
fcns:
{functions}
action:
    var x = {featurizer};
    var row = new(Row, values: x);

    // trees activations
    var activations = a.map(classesTrees, fcn(trees: array(TreeNode) -> double) {{
        var scores = a.map(trees, fcn(tree: TreeNode -> double) {{
          model.tree.simpleWalk(row, tree, fcn(d: Row, t: TreeNode -> boolean) {{
            d.values[t.feature] <= t.value
          }})
        }});
        learningRate * a.sum(scores)
    }});

    // add priors
    activations = la.add(priors, activations);

    // probabilities
    var norm = a.logsumexp(activations);
    var probs = a.map(activations, fcn(x: double -> double) m.exp(x - norm));

    classes[a.argmax(probs)]
    """.format(
        input_record=input_record, featurizer=featurizer, functions=_functions()
    ).strip()

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


def _input_record(types):
    s = 'record(Input'
    for name, typ in types:
        s += ',\n    {}: {}'.format(name, typ)
    return s + '\n);'


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


def _functions():
    return """
    arr = fcn(x: double -> array(double))
        new(array(double), x);

    C = fcn(x: string, categories: array(string) -> array(double))
        a.map(categories, fcn(cat: string -> double) if(cat == x) 1 else 0);

    standardize = fcn(x: double, mu: double, sigma: double -> double)
        (x - mu) / sigma;
""".strip()
