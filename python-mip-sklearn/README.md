[![DockerHub](https://img.shields.io/badge/docker-hbpmip%2Fpython--mip-008bb8.svg)](https://hub.docker.com/r/hbpmip/python-mip-sklearn/)
[![ImageVersion](https://images.microbadger.com/badges/version/hbpmip/python-mip-sklearn.svg)](https://hub.docker.com/r/hbpmip/python-mip-sklearn/tags "hbpmip/python-mip-sklearn image tags")
[![ImageLayers](https://images.microbadger.com/badges/image/hbpmip/python-mip-sklearn.svg)](https://microbadger.com/#/images/hbpmip/python-mip-sklearn "hbpmip/python-mip-sklearn on microbadger")

# Python MIP sklearn

This is a base image for MIP algorithms with additional support for scikit-learn models. It implements additional
models that are not available in the standard library (e.g. `MixedNB`) and contains library `sklearn_to_pfa` for
converting scikit-learn models into PFA.


## Build (for contributors)

Run: `./build.sh`


## Integration Test (for contributors)

Run: `./tests/test.sh`


## Unit Tests (for contributors)

Run: `py.test`


## Publish (for contributors)

Run: `./publish.sh`

## Integrate a new algorithm (for end-users)

1. Extend a version of hbpmip/python-mip-sklearn image (see Dockerfile example below);
2. Add `from mip_helper import mip_helper` in your Python script to import the library;
3. Call `mip_helper.fetch_data()` to get the input data (formatted like described below);
4. Call `mip_helper.save_results(pfa, error, shape)` to store the results;
5. Ensure that your Dockerfile copies the needed files, sets up a valid entry-point
and sets up a _FUNCTION_ environment variable containing the algorithm name (see Dockerfile example below).

### Dockerfile example

```
FROM hbpmip/python-mip-sklearn:0.0.1

ENV FUNCTION python-sgdregress

COPY requirements.txt /requirements.txt
COPY sgd_regression.py /src/sgd_regression.py

RUN conda install -y --file /requirements.txt

ENTRYPOINT ["python", "/src/sgd_regression.py"]
```

