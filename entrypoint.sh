#!/bin/bash
set -e

if [ "$1" == "jupyter" ]; then
	conda run --no-capture-output -n molli jupyter notebook --allow-root --ip=0.0.0.0
else
	conda run --no-capture-output -n molli python ./ncsa-testing/main.py
fi
