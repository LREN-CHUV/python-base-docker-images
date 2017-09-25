FROM hbpmip/python-base

MAINTAINER mirco.nasuti@chuv.ch

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="hbpmip/python-mip" \
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

ENV COMPUTE_IN=/data/in \
    COMPUTE_OUT=/data/out

COPY scripts/io_helper/io_helper.py /
COPY scripts/io_helper/requirements.txt /
COPY scripts/docker-entrypoint.sh /

RUN conda install --yes --file /requirements.txt
RUN conda update --yes --file /requirements.txt

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["compute"]
