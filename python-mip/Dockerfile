FROM hbpmip/python-base

MAINTAINER mirco.nasuti@chuv.ch

LABEL eu.humanbrainproject.category=Python

RUN conda install -y psycopg2
RUN conda update psycopg2

ADD ./database-connector/database-connector.py /
