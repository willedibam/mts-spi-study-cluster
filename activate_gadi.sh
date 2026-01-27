#!/bin/bash
module load python3/3.11.7
module load java/jdk-17.0.2
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
