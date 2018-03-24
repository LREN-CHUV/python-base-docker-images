FROM hbpmip/python-mip:0.4.0

COPY requirements.txt /
COPY docker-entrypoint.sh /

RUN pip install -r /requirements.txt

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

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["compute"]
