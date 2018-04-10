[![DockerHub](https://img.shields.io/badge/docker-hbpmip%2Fpython--mip-008bb8.svg)](https://hub.docker.com/r/hbpmip/python-mip/)
[![ImageVersion](https://images.microbadger.com/badges/version/hbpmip/python-mip.svg)](https://hub.docker.com/r/hbpmip/python-mip/tags "hbpmip/python-mip image tags")
[![ImageLayers](https://images.microbadger.com/badges/image/hbpmip/python-mip.svg)](https://microbadger.com/#/images/hbpmip/python-mip "hbpmip/python-mip on microbadger")

# Python MIP

This is a base image for MIP algorithms implemented in Python.


## Build (for contributors)

Run: `./build.sh`


## Test (for contributors)

Run: `./tests/test.sh`


## Publish (for contributors)

Run: `./publish.sh`


## Integrate a new algorithm (for end-users)

1. Extend a version of hbpmip/python-mip image (see Dockerfile example below);
2. Add `from mip_helper import mip_helper` in your Python script to import the library;
3. Call `mip_helper.fetch_data()` to get the input data (formatted like described below);
4. Call `mip_helper.save_results(pfa, error, shape)` to store the results;
5. Ensure that your Dockerfile copies the needed files, sets up a valid entry-point
and sets up a _FUNCTION_ environment variable containing the algorithm name (see Dockerfile example below).


### Input format

The `mip_helper.fetch_data()` function returns a dictionary.

Here is a complete example:

```
{
  "data": {
    "dependent": [
      {
        "name": "alzheimerbroadcategories",
        "type": {
          "name": "polynominal",
          "enumeration": [
            "AD",
            "CN",
            "Other"
          ]
        },
        "series": [
          "AD",
          "CN",
          "CN"
        ]
      }
    ],
    "independent": [
      {
        "name": "rigthhippocampus",
        "type": {
          "name": "real"
        },
        "series": [
          1.9,
          5.6,
          4.2
        ]
      },
      {
        "name": "lefthippocampus",
        "type": {
          "name": "real"
        },
        "series": [
          1.9,
          5.6,
          4.2
        ]
      }
    ]
  },
  "parameters": [
    {
      "name": "k",
      "value": 42
    }
  ]
}
```

### Dockerfile example

```
FROM hbpmip/python-mip:0.5.3

ENV FUNCTION python-anova

COPY requirements.txt /requirements.txt
COPY anova.py /anova.py

RUN conda install -y --file /requirements.txt

ENTRYPOINT ["python", "/anova.py"]
```


## TODO (for contributors)

Here is a list of future improvements:

* Extract the scripts in a proper library sub-project, publish it on PyPI and install it properly in the python-mip image
* Provide separate images for the building (based on Debian and including tools like conda) and for the distribution (based on Alpine and including a minimal Python environment).
* Add some tests
