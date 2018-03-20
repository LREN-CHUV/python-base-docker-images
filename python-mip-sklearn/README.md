[![DockerHub](https://img.shields.io/badge/docker-hbpmip%2Fpython--mip-008bb8.svg)](https://hub.docker.com/r/hbpmip/python-mip-sklearn/)
[![ImageVersion](https://images.microbadger.com/badges/version/hbpmip/python-mip-sklearn.svg)](https://hub.docker.com/r/hbpmip/python-mip-sklearn/tags "hbpmip/python-mip-sklearn image tags")
[![ImageLayers](https://images.microbadger.com/badges/image/hbpmip/python-mip-sklearn.svg)](https://microbadger.com/#/images/hbpmip/python-mip-sklearn "hbpmip/python-mip-sklearn on microbadger")

# Python MIP sklearn

This is a base image for MIP algorithms with additional support for scikit-learn models. It implements additional
models that are not available in the standard library (e.g. `MixedNB`) and contains library `sklearn_to_pfa` for
converting scikit-learn models into PFA.

For more details about extending this image see
[python-mip](https://github.com/LREN-CHUV/python-base-docker-images/blob/master/python-mip/README.md).


## Build (for contributors)

Run: `./build.sh`


## Integration Test (for contributors)

Run: `./tests/test.sh`


## Unit Tests (for contributors)

Run: `py.test`


## Publish (for contributors)

Run: `./publish.sh`
