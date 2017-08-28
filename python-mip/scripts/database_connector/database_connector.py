#!/usr/bin/env python3

import logging
import psycopg2
import os
import datetime
import re
import json
from urllib.parse import urlparse


'''
************************************************************************************************************************
Initialisation
************************************************************************************************************************
'''

# Configure logging
logging.basicConfig(level=logging.INFO)


# Parse input environment variables for the input DB
postgresql_url = urlparse(os.environ['IN_JDBC_URL']).path
parsed_url = urlparse(postgresql_url)
m = re.search('(.*):([0-9]*)', parsed_url.netloc)
input_db_host = m.group(1)
input_db_port = m.group(2)
m = re.search('/*(.*)', parsed_url.path)
input_db_name = m.group(1)
input_db_user = os.environ['IN_JDBC_USER']
input_db_password = os.environ['IN_JDBC_PASSWORD']


# Parse input environment variables for the output DB, also called analytics DB
postgresql_url = urlparse(os.environ['OUT_JDBC_URL']).path
parsed_url = urlparse(postgresql_url)
m = re.search('(.*):([0-9]*)', parsed_url.netloc)
analytics_db_host = m.group(1)
analytics_db_port = m.group(2)
m = re.search('/*(.*)', parsed_url.path)
analytics_db_name = m.group(1)
analytics_db_user = os.environ['OUT_JDBC_USER']
analytics_db_password = os.environ['OUT_JDBC_PASSWORD']


# Parse metadata environment variable
metadata = json.loads(os.environ['PARAM_meta'])


'''
************************************************************************************************************************
Functions read and write databases
************************************************************************************************************************
'''


def fetch_data():
    """
    Fetch data from science-db using the SQL query given through the PARAM_query environment variable
    :return: A dict containing the columns names 'columns' (see psycopg2 'cursor.description') as a list of strings
    and a list of tuple 'data' where each list element represents a database row and the tuple elements match the
    database columns.
    """
    conn = psycopg2.connect(host=input_db_host, port=input_db_port, dbname=input_db_name, user=input_db_user,
                            password=input_db_password)
    cur = conn.cursor()
    query = os.environ['PARAM_query']
    try:
        cur.execute(query)
        columns = [d.name for d in cur.description]
        data = cur.fetchall()
    except psycopg2.ProgrammingError:
        logging.warning("Cannot execute the following query on the input DB: %s", query)
        columns = []
        data = []
    conn.close()
    return {'columns': columns, 'data': data}


def var_type(var):
    """
    Get variable type and available values if it's a nominal one
    :param var: Variable code as a string
    :return: A dictionary containing the variable type as a string (key 'type')
    and the available values as a list of string (key 'values')
    """
    try:
        var_meta = metadata[var]
    except KeyError:
        logging.warning("Cannot read meta-data for variable %s !", var)
        var_meta = {'type': 'unknown', 'enumerations': []}
    return {
        'type': var_meta['type'] if 'type' in var_meta else 'unknown',
        'values': [e['code'] for e in var_meta['enumerations']] if 'enumerations' in var_meta else []
    }


def save_results(pfa, error, shape):
    """
    Store algorithm results into the analytics-db.
    :param pfa: PFA formated results
    :param error: Error
    :param shape: Result shape. For example pfa_json or pfa_yaml.
    """
    conn = psycopg2.connect(host=analytics_db_host, port=analytics_db_port, dbname=analytics_db_name,
                            user=analytics_db_user, password=analytics_db_password)
    cur = conn.cursor()
    sql = "INSERT INTO job_result VALUES(%s, %s, %s, %s, %s, %s, %s);"
    data = (os.environ['JOB_ID'], os.environ['NODE'], datetime.datetime.now(), pfa, error, shape,
            os.environ['FUNCTION'],)
    cur.execute(sql, data)
    conn.commit()
    conn.close()


'''
************************************************************************************************************************
Wrapper functions to read environment variables
************************************************************************************************************************
'''


def get_job_id():
    """
    Get job ID
    :return: The job ID as a string
    """
    return os.environ['JOB_ID']


def get_node():
    """
    Get data source node
    :return: The node name as a string
    """
    return os.environ['NODE']


def get_docker_image():
    """
    Get Docker image name
    :return: The Docker image name as a string
    """
    return os.environ['DOCKER_IMAGE']


def get_query():
    """
    Get the SQL auto-generated SQL query to get input data
    :return: The SQL query as a string
    """
    return os.environ['PARAM_query']


def get_var():
    """
    Get the variable
    :return: The variable as a string
    """
    return os.environ['PARAM_variables']


def get_covars():
    """
    Get the co-variables
    :return: The list of co-variables as a comma-separated elements string
    """
    return list(filter(None, re.split(', |,', os.environ['PARAM_covariables'])))


def get_gvars():
    """
    Get the grouping variables
    :return: The list of grouping variables as a comma-separated elements string
    """
    return list(filter(None, re.split(', |,', os.environ['PARAM_grouping'])))


def get_code():
    """
    Get the algorithm code name
    :return: The algorithm code name as a string
    """
    return os.environ['CODE']


def get_name():
    """
    Get the algorithm name
    :return: The algorithm name as a string
    """
    return os.environ['NAME']


def get_model():
    """
    Get the model name i.e. the output type
    :return: The model name as a string
    """
    return os.environ['MODEL']


def get_function():
    """
    Get the function name
    :return: The function name as a string
    """
    return os.environ['FUNCTION']


def get_parameter(p):
    """
    Get the function parameter given in argument
    :return: The parameter value
    """
    return json.loads(os.environ['PARAM_MODEL_' + p])
