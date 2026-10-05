#!/bin/bash

uvicorn api.app:app --host 127.0.0.1 --port 8001 &

streamlit run frontend.py \
  --server.address 0.0.0.0 \
  --server.port "$DATABRICKS_APP_PORT"