from mip_helper.mip_helper.io_helper import _format_variable
from . import fixtures as fx


def test_format_variable():
    r = _format_variable('lefthippocampus', fx.data().to_dict('list'), fx.metadata())
    del r['series']
    assert r == {'name': 'lefthippocampus', 'type': {'name': 'real'}, 'mean': 3.0, 'std': 0.35}

    r = _format_variable('agegroup', fx.data().to_dict('list'), fx.metadata())
    assert r == {
        'name': 'agegroup',
        'type': {
            'name': 'polynominal',
            'enumeration': ['-50y', '50-59y', '60-69y', '70-79y', '+80y']
        },
        'series': ['70-79y', '70-79y', '70-79y', '70-79y', '+80y']
    }
