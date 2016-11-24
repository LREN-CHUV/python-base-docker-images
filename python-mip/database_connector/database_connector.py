#!/usr/bin/env python3

import psycopg2
import os
import datetime
import re
from urllib.parse import urlparse


'''
************************************************************************************************************************
Initialisation
************************************************************************************************************************
'''


# Parse input environment variables for science-db
postgresql_url = urlparse(os.environ['IN_JDBC_URL']).path
parsed_url = urlparse(postgresql_url)
m = re.search('(.*):([0-9]*)', parsed_url.netloc)
science_db_host = m.group(1)
science_db_port = m.group(2)
m = re.search('/*(.*)', parsed_url.path)
science_db_name = m.group(1)
science_db_user = os.environ['IN_JDBC_USER']
science_db_password = os.environ['IN_JDBC_PASSWORD']


# Parse input environment variables for analytics-db
postgresql_url = urlparse(os.environ['OUT_JDBC_URL']).path
parsed_url = urlparse(postgresql_url)
m = re.search('(.*):([0-9]*)', parsed_url.netloc)
analytics_db_host = m.group(1)
analytics_db_port = m.group(2)
m = re.search('/*(.*)', parsed_url.path)
analytics_db_name = m.group(1)
analytics_db_user = os.environ['OUT_JDBC_USER']
analytics_db_password = os.environ['OUT_JDBC_PASSWORD']


'''
************************************************************************************************************************
Functions read and write databases
************************************************************************************************************************
'''


def fetch_data():
    """
    Fetch data from science-db using the SQL query given through the PARAM_query environment variable
    :return: A list of tuple where each list element represents a database row
    and the tuple elements match the database columns.
    """
    conn = psycopg2.connect(host=science_db_host, port=science_db_port, dbname=science_db_name, user=science_db_user,
                            password=science_db_password)
    cur = conn.cursor()
    cur.execute(os.environ['PARAM_query'])
    data = cur.fetchall()
    return data


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
    cur.execute("""INSERT INTO job_result VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s')"""
                % (os.environ['JOB_ID'], os.environ['NODE'], datetime.datetime.now(), pfa, error, shape,
                   os.environ['FUNCTION']))
    conn.commit()
    conn.close()


'''
************************************************************************************************************************
Wrapper functions to read environment variables
************************************************************************************************************************
'''


def get_job_id():
    return os.environ['JOB_ID']


def get_node():
    return os.environ['NODE']


def get_docker_image():
    return os.environ['DOCKER_IMAGE']


def get_query():
    return os.environ['PARAM_query']


def get_var():
    return os.environ['PARAM_variables']


def get_covars():
    return os.environ['PARAM_covariables']


def get_gvars():
    return os.environ['PARAM_grouping']


def get_code():
    return os.environ['CODE']


def get_name():
    return os.environ['NAME']


def get_model():
    return os.environ['MODEL']


def get_function():
    return os.environ['FUNCTION']
