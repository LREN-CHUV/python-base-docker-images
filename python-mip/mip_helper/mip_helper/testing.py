#!/usr/bin/env python3
"""
Utility functions for testing.
"""

import copy
import numpy as np

SEED = 42


def round_dict(d, precision=3):
    """Round all numerical values in a dictionary recursively."""
    d = copy.deepcopy(d)
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                d[k] = round(v, precision)
            except TypeError:
                d[k] = round_dict(v)
        return d
    elif isinstance(d, list):
        return [round_dict(v) for v in d]
    elif isinstance(d, tuple):
        return tuple([round_dict(v) for v in d])
    elif isinstance(d, float):
        return round(d, precision)

    return d


def random_real_series(x, add_null=False, limit_from=0, limit_to=5, seed=SEED):
    np.random.seed(seed)
    s = np.random.normal(x['mean'], x['std'], size=limit_to)
    s = np.minimum(np.maximum(s, x['minValue']), x['maxValue'])
    if add_null and len(s):
        s[np.random.choice(limit_to)] = None
    return list(s)[limit_from:]


def random_integer_series(x, **kwargs):
    s = random_real_series(x, **kwargs)
    return [int(e) if e is not None else None for e in s]


def random_nominal_series(x, add_null=False, limit_from=0, limit_to=5, seed=SEED):
    np.random.seed(seed)
    s = np.random.choice(x['type']['enumeration'], size=limit_to)
    if add_null and len(s):
        s[np.random.choice(limit_to)] = None
    return list(s)[limit_from:]


def independent(include_real=True, include_integer=True, include_nominal=False, **kwargs):
    if 'add_independent_null' in kwargs:
        kwargs['add_null'] = kwargs.pop('add_independent_null')

    ret = []
    if include_real:
        x = {
            'name': 'subjectage',
            'type': {
                'name': 'real'
            },
            'series': [],
            'mean': 70.4,
            'std': 8.3,
            'minValue': 30.,
            'maxValue': 90.,
            'label': 'Exact age'
        }
        x['series'] = random_real_series(x, seed=1, **kwargs)
        ret.append(x)

    if include_integer:
        x = {
            'name': 'minimentalstate',
            'type': {
                'name': 'integer'
            },
            'series': [],
            'mean': 24.4,
            'std': 5.2,
            'minValue': 0,
            'maxValue': 30,
            'label': 'MMSE Total scores'
        }
        x['series'] = random_integer_series(x, seed=2, **kwargs)
        ret.append(x)

    if include_nominal:
        x = {
            'name': 'agegroup',
            'type': {
                'name': 'polynominal',
                'enumeration': ['-50y', '50-59y']
            },
            'label': 'Age Group',
            'series': []
        }
        x['series'] = random_nominal_series(x, seed=3, **kwargs)
        ret.append(x)

    return ret


def inputs_regression(add_null=False, limit_from=0, limit_to=5, **kwargs):
    x = {
        'name': 'lefthippocampus',
        'label': 'Left Hippocampus',
        'type': {
            'name': 'real'
        },
        'series': [],
        'mean': 3.,
        'std': 0.39,
        'minValue': 1.,
        'maxValue': 5.,
    }
    x['series'] = random_real_series(x, seed=4, add_null=add_null, limit_from=limit_from, limit_to=limit_to)
    return {
        'data': {
            'dependent': [x],
            'independent': independent(limit_from=limit_from, limit_to=limit_to, **kwargs)
        },
        'parameters': []
    }


def inputs_classification(add_null=False, limit_from=0, limit_to=5, **kwargs):
    x = {
        'name': 'adnicategory',
        'label': 'ADNI category',
        'type': {
            'name': 'polynominal',
            'enumeration': ['AD', 'CN', 'Other'],
            'enumeration_labels': ['Alzheimers disease', 'Cognitively Normal', 'Other']
        },
        'series': []
    }
    x['series'] = random_nominal_series(x, seed=5, add_null=add_null, limit_from=limit_from, limit_to=limit_to)
    return {
        'data': {
            'dependent': [x],
            'independent': independent(limit_from=limit_from, limit_to=limit_to, **kwargs)
        },
        'parameters': []
    }
