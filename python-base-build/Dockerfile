FROM continuumio/miniconda3:4.3.27

MAINTAINER mirco.nasuti@chuv.ch

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

COPY requirements-dev.txt /

RUN pip install -r /requirements-dev.txt

COPY requirements.txt /

RUN pip install -r /requirements.txt

# Light version of Titus for Python3 installed from Git
# LATER: install Titus for Python 3 from Pypi once it is released officially
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /tmp/* /var/tmp/* /var/cache/* /var/lib/apt/lists/*

RUN pip install git+https://github.com/Marigold/hadrian.git@light#subdirectory=titus

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="hbpmip/python-base-build" \
      org.label-schema.description="Base Python image including Conda" \
      org.label-schema.url="https://github.com/LREN-CHUV/python-base-docker-images" \
      org.label-schema.vcs-type="git" \
      org.label-schema.vcs-url="https://github.com/LREN-CHUV/python-base-docker-images.git" \
      org.label-schema.vcs-ref="$VCS_REF" \
      org.label-schema.version="$VERSION" \
      org.label-schema.vendor="LREN CHUV" \
      org.label-schema.license="Apache2.0" \
      org.label-schema.docker.dockerfile="Dockerfile" \
      org.label-schema.schema-version="1.0"
