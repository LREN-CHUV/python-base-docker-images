[![CHUV](https://img.shields.io/badge/CHUV-LREN-AF4C64.svg)](https://www.unil.ch/lren/en/home.html) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/LREN-CHUV/python-base-docker-images/blob/master/LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/495d61eb1adf4445a822c57c516ab892)](https://www.codacy.com/app/hbp-mip/python-base-docker-images?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LREN-CHUV/python-base-docker-images&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/HBPMedical/python-base-docker-images.svg?style=svg)](https://circleci.com/gh/HBPMedical/python-base-docker-images)

# Python Base Docker Images

## What is it ?

This is a collection of Docker images for Python.


## Images

### [hbpmip/python-base](python-base)

Minimal Python 3.x image.


### [hbpmip/python-base-build](python-base-build)

Python 3.x image including Miniconda (so you can use conda to install dependencies).
Python algorithms can be built using this image !


### [hbpmip/python-mip](python-mip)

Python 3.x image based on python-base and providing extra libraries like pandas and
a library to read/write MIP algorithms inputs/outputs.
Python algorithms should be distributed as children images of this one.
