FROM hbpmip/python-base

MAINTAINER mirco.nasuti@chuv.ch

LABEL eu.humanbrainproject.category=Python

ENV COMPUTE_IN=/data/in \
    COMPUTE_OUT=/data/out

RUN conda install -y psycopg2
RUN conda update psycopg2

COPY scripts/database_connector/database_connector.py /
COPY scripts/docker-entrypoint.sh /
