import mock
import json
import os
import numpy as np
import pandas as pd
import datetime
from contextlib import contextmanager
from freezegun import freeze_time
from mip_helper.io_helper import _format_variable, save_results, save_error, fetch_data, fetch_dataframe
from mip_helper.shapes import Shapes
from . import fixtures as fx

META = json.dumps(
    {
        "lefthippocampus": {
            "code": "lefthippocampus",
            "type": "real",
            "mean": 3.0,
            "std": 0.35,
            "minValue": 0.1,
            "maxValue": 5.0
        },
        "subjectageyears": {
            "code": "subjectageyears",
            "label": "Age Years",
            "maxValue": 130,
            "minValue": 0,
            "type": "integer"
        }
    }
)

ENVIRON = {
    'JOB_ID': '1',
    'NODE': 'test',
    'FUNCTION': 'unit-test',
    'OUT_DBAPI_DRIVER': 'postgresql',
    'OUT_HOST': 'db',
    'OUT_PORT': '5432',
    'OUT_DATABASE': 'postgres',
    'OUT_USER': 'postgres',
    'OUT_PASSWORD': 'pwd',
    'IN_DBAPI_DRIVER': 'postgresql',
    'IN_HOST': 'db',
    'IN_PORT': '5432',
    'IN_DATABASE': 'postgres',
    'IN_USER': 'postgres',
    'IN_PASSWORD': 'pwd',
    'PARAM_query': 'SELECT lefthippocampus, subjectageyears FROM features',
    'PARAM_variables': 'lefthippocampus',
    'PARAM_covariables': 'subjectageyears',
    'PARAM_meta': META,
}

PARAMETERS = json.dumps(
    {
        "query": "SELECT lefthippocampus, subjectageyears FROM features",
        "variables": "lefthippocampus",
        "covariables": ["subjectageyears"],
        "model_parameters": {}
    }
)


@contextmanager
def mock_engine():
    with mock.patch.dict(os.environ, ENVIRON):
        with mock.patch('sqlalchemy.create_engine') as mock_create_engine:
            with freeze_time('2018-01-01'):
                engine = mock.MagicMock()
                mock_create_engine.return_value = engine
                yield engine


@mock.patch('pandas.read_sql_query')
def test_fetch_data(mock_read_sql_query):
    data = pd.DataFrame({
        'lefthippocampus': [1., 2.],
        'subjectageyears': [20, 30],
    })
    mock_read_sql_query.return_value = data

    with mock_engine():
        inputs = fetch_data()

    assert inputs == {
        'data': {
            'dependent': [
                {
                    'label': 'lefthippocampus',
                    'mean': 3.0,
                    'name': 'lefthippocampus',
                    'series': [1.0, 2.0],
                    'std': 0.35,
                    'type': {
                        'name': 'real'
                    }
                }
            ],
            'independent':
            [{
                'label': 'Age Years',
                'name': 'subjectageyears',
                'series': [20, 30],
                'type': {
                    'name': 'integer'
                }
            }]
        },
        'parameters': []
    }


@mock.patch('pandas.read_sql_query')
def test_fetch_dataframe(mock_read_sql_query):
    data = pd.DataFrame({
        'lefthippocampus': [1., 2.],
        'subjectageyears': [20, 30],
    })
    mock_read_sql_query.return_value = data

    with mock_engine():
        df = fetch_dataframe()

    assert df.to_dict(orient='records') == [
        {
            'subjectageyears': 20.0,
            'lefthippocampus': 1.0
        }, {
            'subjectageyears': 30.0,
            'lefthippocampus': 2.0
        }
    ]


def test_save_results():
    results = json.dumps({'a': 'b'})

    with mock_engine() as engine:
        save_results(results, Shapes.JSON)

    assert engine.execute.call_args[1] == {
        'job_id': '1',
        'node': 'test',
        'timestamp': datetime.datetime(2018, 1, 1, 0, 0),
        'data': '{"a": "b"}',
        'error': None,
        'shape': 'application/json',
        'function': 'unit-test',
        'result_name': '',
        'result_title': None,
        'parameters': PARAMETERS,
    }


def test_save_error():
    error = ValueError('mytest')

    with mock_engine() as engine:
        save_error(error)

    assert engine.execute.call_args[1] == {
        'job_id': '1',
        'node': 'test',
        'timestamp': datetime.datetime(2018, 1, 1, 0, 0),
        'error': 'mytest',
        'shape': 'text/plain+error',
        'function': 'unit-test',
        'parameters': PARAMETERS
    }


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
