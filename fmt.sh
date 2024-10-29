#!/bin/bash
echo "Running fmt"
echo "Running isort"
isort . --profile black
echo "Running black"
black .
