# Python Base Docker Images

## What is it ?

This is a collection of Docker images for Python.


## Images

* python-base : Python 3 (includes Conda)
* python-mip : python-base + sqlalchemy + pandas + custom library to easily read/write algorithms inputs/outputs


## Usage

### Build

Run: `./build.sh`


### Integrate a new algorithm

1. Extend a version of hbpmip/python-mip image;
2. Add `import io_helper` in your Python script to import the library;
3. Call `io_helper.fetch_data()` to get the input data (formatted like described below);
4. Call `io_helper.save_results(pfa, error, shape)` to store the results;
5. Ensure that your Dockerfile copies the needed files and sets up a valid entry-point.


## Input format

The `io_helper.fetch_data()` function returns a dictionary.

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

## Dockerfile example

```
FROM hbpmip/python-mip:1.0.0

MAINTAINER mirco.nasuti@chuv.ch

COPY requirements.txt /requirements.txt
COPY anova.py /anova.py

RUN conda install -y --file /requirements.txt

ENTRYPOINT ["python", "/anova.py"]
```
