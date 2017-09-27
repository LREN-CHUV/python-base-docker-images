[![DockerHub](https://img.shields.io/badge/docker-hbpmip%2Fpython--base-008bb8.svg)](https://hub.docker.com/r/hbpmip/python-base/)
[![ImageVersion](https://images.microbadger.com/badges/version/hbpmip/python-base.svg)](https://hub.docker.com/r/hbpmip/python-base/tags "hbpmip/python-base image tags")
[![ImageLayers](https://images.microbadger.com/badges/image/hbpmip/python-base.svg)](https://microbadger.com/#/images/hbpmip/python-base "hbpmip/python-base on microbadger")

# Python Base

This is a base image for Python.

## Usage

Dockerfile
```
  FROM hbpmip/python-base:0.2.0

```

## Build

Run: `./build.sh`


## Test

Run: `./tests/test.sh`


## Publish

Run: `./publish.sh`


## TODO

* Provide separate images for the building (based on Debian and including tools like conda) and for the distribution (based on Alpine and including a minimal Python environment).
* Add some tests
