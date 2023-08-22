#!/bin/bash

set -e

if ! curl -s --head  --request GET http://localhost:8501/ | grep "200 OK" > /dev/null; then
    echo "Streamlit is not up"
    exit 1
fi

if ! curl -s --head  --request GET http://localhost:7474/ | grep "200 OK" > /dev/null; then
    echo "Neo4j is not up"
    exit 1
fi

if ! curl -s --head  --request GET http://localhost:3000/ | grep "200 OK" > /dev/null; then
    echo "Grafana is not up"
    exit 1
fi


