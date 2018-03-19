FROM hbpmip/python-base:0.3.0

MAINTAINER mirco.nasuti@chuv.ch

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

COPY requirements.txt /
COPY docker-entrypoint.sh /

RUN pip install -r /requirements.txt

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["compute"]

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="hbpmip/python-mip" \
      org.label-schema.description="Execute Python algorithms in MIP Woken environment" \
      org.label-schema.url="https://github.com/LREN-CHUV/python-base-docker-images" \
      org.label-schema.vcs-type="git" \
      org.label-schema.vcs-url="https://github.com/LREN-CHUV/python-base-docker-images.git" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.version="$VERSION" \
      org.label-schema.vendor="LREN CHUV" \
      org.label-schema.license="Apache2.0" \
      org.label-schema.docker.dockerfile="Dockerfile" \
      org.label-schema.schema-version="1.0"
