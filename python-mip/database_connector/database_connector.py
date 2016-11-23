#!/usr/bin/env python3

import psycopg2
import os


def get_vars_metadata():
    conn = psycopg2.connect(host=os.environ['META_DB_HOST'], port=os.environ['META_DB_PORT'],
                            dbname=os.environ['META_DB_NAME'], user=os.environ['META_DB_USER'],
                            password=os.environ['META_DB_PASSWORD'])
    cur = conn.cursor()
    cur.execute("""SELECT hierarchy FROM meta_variables""")
    metadata = cur.fetchone()[0]
    conn.close()
    return metadata


def get_vars_names():
    conn = psycopg2.connect(host=os.environ['SCIENCE_DB_HOST'], port=os.environ['SCIENCE_DB_PORT'],
                            dbname=os.environ['SCIENCE_DB_NAME'], user=os.environ['SCIENCE_DB_USER'],
                            password=os.environ['SCIENCE_DB_PASSWORD'])
    cur = conn.cursor()
    cur.execute("""SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = 'adni_merge'""")
    names = [t[0] for t in cur.fetchall()]
    conn.close()
    return names


def get_vars_data():
    conn = psycopg2.connect(host=os.environ['SCIENCE_DB_HOST'], port=os.environ['SCIENCE_DB_PORT'],
                            dbname=os.environ['SCIENCE_DB_NAME'], user=os.environ['SCIENCE_DB_USER'],
                            password=os.environ['SCIENCE_DB_PASSWORD'])
    cur = conn.cursor()
    cur.execute("""SELECT * FROM adni_merge""")
    data = cur.fetchall()
    return data


def save_results(job_id, node, timestamp, pfa, error, shape, function):
    conn = psycopg2.connect(host=os.environ['ANALYTICS_DB_HOST'], port=os.environ['ANALYTICS_DB_PORT'],
                            dbname=os.environ['ANALYTICS_DB_NAME'], user=os.environ['ANALYTICS_DB_USER'],
                            password=os.environ['ANALYTICS_DB_PASSWORD'])
    cur = conn.cursor()
    cur.execute("""INSERT INTO job_result VALUES('%s', '%s', '%s', '%s', '%s', '%s', '%s')"""
                % (job_id, node, timestamp, pfa, error, shape, function))
    conn.commit()
    conn.close()
