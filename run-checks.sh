#!/bin/bash
set -e

flake8
black --check .
isort --check-only .
mypy .

pytest -vvs --cov=llm_demo --cov-report term-missing
