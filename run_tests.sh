#!/bin/bash
pytest tests \
  -v \
  --html=build/tests/report.html \
  --junitxml=build/tests/report.xml \
  --cov=revu \
  --cov-report=term \
  --cov-report=html:build/tests/coverage
