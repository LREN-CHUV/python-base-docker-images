#!/usr/bin/env python3

import psycopg2


# Default values for databases configuration

science_db_host = '172.17.0.1'
science_db_port = '65432'
science_db_name = 'science'
science_db_user = 'science'
science_db_password = 'sciencepass'

meta_db_host = '172.17.0.1'
meta_db_port = '65432'
meta_db_name = 'meta'
meta_db_user = 'meta'
meta_db_password = 'metapass'

analytics_db_host = '172.17.0.1'
analytics_db_port = '65432'
analytics_db_name = 'analytics'
analytics_db_user = 'analytics'
analytics_db_password = 'analyticspass'


def setup_science_db(host, port, name, user, password):
    global science_db_host
    global science_db_port
    global science_db_name
    global science_db_user
    global science_db_password

    science_db_host = host
    science_db_port = port
    science_db_name = name
    science_db_user = user
    science_db_password = password


def setup_meta_db(host, port, name, user, password):
    global meta_db_host
    global meta_db_port
    global meta_db_name
    global meta_db_user
    global meta_db_password

    meta_db_host = host
    meta_db_port = port
    meta_db_name = name
    meta_db_user = user
    meta_db_password = password


def setup_analytics_db(host, port, name, user, password):
    global analytics_db_host
    global analytics_db_port
    global analytics_db_name
    global analytics_db_user
    global analytics_db_password

    analytics_db_host = host
    analytics_db_port = port
    analytics_db_name = name
    analytics_db_user = user
    analytics_db_password = password


def get_vars_metadata():
    conn = psycopg2.connect(host=meta_db_host, port=meta_db_port, dbname=meta_db_name, user=meta_db_user,
                            password=meta_db_password)
    cur = conn.cursor()
    cur.execute("""SELECT hierarchy FROM meta_variables""")
    metadata = cur.fetchone()[0]
    conn.close()
    return metadata


def get_vars_names():
    conn = psycopg2.connect(host=science_db_host, port=science_db_port, dbname=science_db_name, user=science_db_user,
                            password=science_db_password)
    cur = conn.cursor()
    cur.execute("""SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = 'adni_merge'""")
    names = [t[0] for t in cur.fetchall()]
    conn.close()
    return names


def get_vars_data():
    conn = psycopg2.connect(host=science_db_host, port=science_db_port, dbname=science_db_name, user=science_db_user,
                            password=science_db_password)
    cur = conn.cursor()
    cur.execute("""SELECT * FROM adni_merge""")
    data = cur.fetchall()
    return data


def save_results(job_id, node, timestamp, pfa, error, shape, function):
    conn = psycopg2.connect(host=analytics_db_host, port=analytics_db_port, dbname=analytics_db_name,
                            user=analytics_db_user, password=analytics_db_password)
    cur = conn.cursor()
    cur.execute("""INSERT INTO job_result VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s')"""
                % (job_id, node, timestamp, pfa, error, shape, function))
    conn.commit()
    conn.close()
