#!/usr/bin/env python3

import logging
import sqlalchemy
import pandas
import os
import datetime
import re
import json
from urllib.parse import urlparse


# *********************************************************************************************************************
# Initialization
# *********************************************************************************************************************

# Configure logging
logging.basicConfig(level=logging.INFO)


# Get input DB information
parsed_in_jdbc_url = urlparse(urlparse(os.environ['IN_JDBC_URL']).path)
in_jdbc_url = parsed_in_jdbc_url.scheme + \
              "://" + os.environ['IN_JDBC_USER'] + ":" + os.environ['IN_JDBC_PASSWORD'] + \
              "@" + parsed_in_jdbc_url.netloc + parsed_in_jdbc_url.path


# Get output DB information
parsed_out_jdbc_url = urlparse(urlparse(os.environ['OUT_JDBC_URL']).path)
out_jdbc_url = parsed_out_jdbc_url.scheme + \
               "://" + os.environ['OUT_JDBC_USER'] + ":" + os.environ['OUT_JDBC_PASSWORD'] + \
               "@" + parsed_out_jdbc_url.netloc + parsed_out_jdbc_url.path


# Get variables meta-data
metadata = json.loads(os.environ['PARAM_meta'])

# Get SQL query
query = os.environ['PARAM_query']

# Get variables code
var = os.environ['PARAM_variables']
covars = list(filter(None, re.split(', |,', os.environ['PARAM_covariables']))) + \
         list(filter(None, re.split(', |,', os.environ['PARAM_grouping'])))


# *********************************************************************************************************************
# Public functions
# *********************************************************************************************************************

def fetch_data():
    """
    Get all the needed  algorithm inputs (data, algorithm parameters, etc).
    The inputs format is described in the README file.
    """
    engine = sqlalchemy.create_engine(in_jdbc_url)
    df = pandas.read_sql_query(query, engine)
    raw_data = df.to_dict('list')

    data = dict()
    data['dependent'] = [_format_variable(var, raw_data, metadata)]
    data['independent'] = [_format_variable(v, raw_data, metadata) for v in covars]

    parameters = _get_parameters()

    inputs = {'data': data, 'parameters': parameters}

    return inputs


def save_results(pfa, error, shape):
    """
    Store algorithm results in the output DB.
    :param pfa: PFA formatted results
    :param error: Error message (if any)
    :param shape: Result shape. For example: pfa_json.
    """
    engine = sqlalchemy.create_engine(out_jdbc_url)

    sql = sqlalchemy.text("INSERT INTO job_result VALUES(:job_id, :node, :date, :pfa, :error, :shape, :function)")
    engine.execute(sql,
                   job_id=os.environ['JOB_ID'],
                   node=os.environ['NODE'],
                   date=datetime.datetime.now(),
                   pfa=pfa,
                   error=error,
                   shape=shape,
                   function=os.environ['FUNCTION'])


# *********************************************************************************************************************
# Private functions
# *********************************************************************************************************************

def _format_variable(var_code, raw_data, vars_meta):
    var_type = _get_type(var_code, vars_meta)
    return {'name': var_code, 'type': var_type, 'series': raw_data[var_code]}


def _get_parameters():
    param_prefix = "PARAM_MODEL_"
    research_pattern = param_prefix + ".*"
    parameters = []
    for env_var in os.environ:
        if re.fullmatch(research_pattern, env_var):
            parameters.append({'name': env_var.split(param_prefix)[1], 'value': os.environ[env_var]})
    return parameters


def _get_type(var_code, vars_meta):
    type_info = dict()
    try:
        var_meta = vars_meta[var_code]
        type_info['name'] = var_meta['type'] if 'type' in var_meta else 'unknown'
        if type_info['name'] in ['polynominal', 'binominal']:
            type_info['enumeration'] = [e['code'] for e in var_meta['enumerations']]
    except KeyError:
        logging.warning("Cannot read meta-data for variable %s !", var_code)
        type_info['name'] = 'unknown'
    return type_info
