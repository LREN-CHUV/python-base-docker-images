FROM hbpmip/python-base

MAINTAINER mirco.nasuti@chuv.ch

LABEL eu.humanbrainproject.category=Python

ENV COMPUTE_IN=/data/in \
    COMPUTE_OUT=/data/out

COPY scripts/database_connector/database_connector.py /
COPY scripts/database_connector/requirements.txt /
COPY scripts/docker-entrypoint.sh /

RUN conda install --yes --file /requirements.txt
RUN conda update --yes --file /requirements.txt

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["compute"]
