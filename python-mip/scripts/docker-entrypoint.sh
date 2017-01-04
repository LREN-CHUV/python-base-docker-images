#!/usr/bin/env bash

if [ "$1" = "compute" ]; then
	mkdir -p "$COMPUTE_IN" "$COMPUTE_OUT"
	chown -R compute "$COMPUTE_IN" "$COMPUTE_OUT"
fi
