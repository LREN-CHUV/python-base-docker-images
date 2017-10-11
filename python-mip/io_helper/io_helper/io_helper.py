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


# *********************************************************************************************************************
# Public functions
# *********************************************************************************************************************

def fetch_data():
    """
    Get all the needed  algorithm inputs (data, algorithm parameters, etc).
    The inputs format is described in the README file.
    """
    engine = sqlalchemy.create_engine(_get_input_jdbc_url())
    df = pandas.read_sql_query(_get_query(), engine)
    raw_data = df.to_dict('list')

    var = _get_var()
    covars = _get_covars()
    metadata = _get_metadata()

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
    engine = sqlalchemy.create_engine(_get_output_jdbc_url())

    sql = sqlalchemy.text("INSERT INTO job_result VALUES(:job_id, :node, :date, :pfa, :error, :shape, :function)")
    engine.execute(sql,
                   job_id=_get_job_id(),
                   node=_get_node(),
                   date=datetime.datetime.now(),
                   pfa=pfa,
                   error=error,
                   shape=shape,
                   function=_get_function())


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


def _get_input_jdbc_url():
    try:
        raw_url = os.environ['IN_JDBC_URL']
    except KeyError:
        logging.warning("Cannot read input JDBC URL from environment variable IN_JDBC_URL")
        raw_url = ""
    try:
        user = os.environ['IN_JDBC_USER']
    except KeyError:
        logging.warning("Cannot read input JDBC user from environment variable IN_JDBC_USER")
        user = ""
    try:
        passwd = os.environ['IN_JDBC_PASSWORD']
    except KeyError:
        logging.warning("Cannot read input JDBC password from environment variable IN_JDBC_PASSWORD")
        passwd = ""
    parsed_in_jdbc_url = urlparse(urlparse(raw_url).path)
    scheme = parsed_in_jdbc_url.scheme
    netloc = parsed_in_jdbc_url.netloc
    path = parsed_in_jdbc_url.path
    return scheme + "://" + user + ":" + passwd + "@" + netloc + path


def _get_output_jdbc_url():
    try:
        raw_url = os.environ['OUT_JDBC_URL']
    except KeyError:
        logging.warning("Cannot read input JDBC URL from environment variable OUT_JDBC_URL")
        raw_url = ""
    try:
        user = os.environ['OUT_JDBC_USER']
    except KeyError:
        logging.warning("Cannot read input JDBC user from environment variable OUT_JDBC_USER")
        user = ""
    try:
        passwd = os.environ['OUT_JDBC_PASSWORD']
    except KeyError:
        logging.warning("Cannot read input JDBC password from environment variable OUT_JDBC_PASSWORD")
        passwd = ""
    parsed_in_jdbc_url = urlparse(urlparse(raw_url).path)
    scheme = parsed_in_jdbc_url.scheme
    netloc = parsed_in_jdbc_url.netloc
    path = parsed_in_jdbc_url.path
    return scheme + "://" + user + ":" + passwd + "@" + netloc + path


def _get_metadata():
    try:
        return json.loads(os.environ['PARAM_meta'])
    except KeyError:
        logging.warning("Cannot read metadata from environment variable PARAM_meta")


def _get_query():
    try:
        return os.environ['PARAM_query']
    except KeyError:
        logging.warning("Cannot read SQL query from environment variable PARAM_query")


def _get_var():
    try:
        return os.environ['PARAM_variables']
    except KeyError:
        logging.warning("Cannot read dependent variables from environment variable PARAM_variables")


def _get_covars():
    try:
        covars = os.environ['PARAM_covariables']
    except KeyError:
        covars = ""
    try:
        gvars = os.environ['PARAM_grouping']
    except KeyError:
        gvars = ""
    return list(filter(None, re.split(', |,', covars))) + list(filter(None, re.split(', |,', gvars)))


def _get_job_id():
    try:
        return os.environ['JOB_ID']
    except KeyError:
        logging.warning("Cannot read job ID from environment variable JOB_ID")


def _get_node():
    try:
        return os.environ['NODE']
    except KeyError:
        logging.warning("Cannot read node from environment variable NODE")


def _get_function():
    try:
        return os.environ['FUNCTION']
    except KeyError:
        logging.warning("Cannot read function from environment variable FUNCTION")
