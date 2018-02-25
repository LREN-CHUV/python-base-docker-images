#!/usr/bin/env bash

CHECK=$(docker run --entrypoint=/bin/bash hbpmip/python-mip -c "pip list --format=legacy | grep -c mip-helper")

if [ ${CHECK} -ne 1 ]; then
    exit 1
fi

exit 0
