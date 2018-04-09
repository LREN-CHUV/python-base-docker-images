FROM hbpmip/python-base-build:0.4.5

COPY sklearn_to_pfa/requirements-dev.txt /requirements-dev.txt
RUN pip install -r /requirements-dev.txt

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY sklearn_to_pfa /sklearn_to_pfa
WORKDIR /sklearn_to_pfa
RUN python setup.py install

WORKDIR /

# Unit tests for sklearn_to_pfa
COPY sklearn_to_pfa/tests /src/tests/

WORKDIR /src
RUN pytest tests/

## Target image

FROM hbpmip/python-mip:0.5.3

COPY requirements.txt /
COPY docker-entrypoint.sh /

RUN pip install -r /requirements.txt

# Light version of Titus for Python3 installed from Git
# LATER: install Titus for Python 3 from Pypi once it is released officially
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /tmp/* /var/tmp/* /var/cache/* /var/lib/apt/lists/*

RUN pip install git+https://github.com/Marigold/hadrian.git@light#subdirectory=titus

# TODO: this is only for prototyping, should be installed from pypi once it gets stable
COPY sklearn_to_pfa /sklearn_to_pfa
WORKDIR /sklearn_to_pfa
RUN python setup.py install

WORKDIR /

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="hbpmip/python-mip-sklearn" \
      org.label-schema.description="Execute Python algorithms in MIP Woken environment" \
      org.label-schema.url="https://github.com/LREN-CHUV/python-base-docker-images" \
      org.label-schema.vcs-type="git" \
      org.label-schema.vcs-url="https://github.com/LREN-CHUV/python-base-docker-images.git" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version="$VERSION" \
      org.label-schema.vendor="LREN CHUV" \
      org.label-schema.license="AGPLv3" \
      org.label-schema.docker.dockerfile="Dockerfile" \
      org.label-schema.schema-version="1.0"

RUN apt-get update
RUN apt-get install -y git

COPY requirements.txt /
COPY docker-entrypoint.sh /

RUN pip install -r /requirements.txt
RUN pip install git+https://github.com/Marigold/hadrian.git@light#subdirectory=titus

# NOTE: this is only for prototyping, should be installed from pypi once it gets stable
COPY sklearn_to_pfa /sklearn_to_pfa
RUN cd sklearn_to_pfa && python setup.py install

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["compute"]
