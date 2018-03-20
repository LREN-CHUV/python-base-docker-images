import numpy as np
from mip_helper.io_helper import _format_variable
from . import fixtures as fx


def test_format_variable():
    r = _format_variable('lefthippocampus', fx.data().to_dict('list'), fx.metadata())
    del r['series']
    assert r == {
        'name': 'lefthippocampus',
        'type': {
            'name': 'real'
        },
        'mean': 3.0,
        'std': 0.35,
        'label': 'lefthippocampus'
    }

    r = _format_variable('agegroup', fx.data().to_dict('list'), fx.metadata())
    assert r == {
        'name': 'agegroup',
        'type': {
            'name': 'polynominal',
            'enumeration': ['-50y', '50-59y', '60-69y', '70-79y', '+80y'],
            'enumeration_labels': ['-50y', '50-59y', '60-69y', '70-79y', '+80y']
        },
        'series': ['70-79y', '70-79y', '70-79y', '70-79y', '+80y'],
        'label': 'Age Group'
    }


def test_format_variable_None_nan():
    data = fx.data().to_dict('list')
    data['lefthippocampus'] = [np.nan, None, 42.]
    r = _format_variable('lefthippocampus', data, fx.metadata())
    assert r['series'] == [None, None, 42.0]
