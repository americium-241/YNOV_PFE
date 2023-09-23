#!/bin/bash
# Give Grafana some time to initialize


echo "Starting Grafana in the background..."
# Start Grafana in the background

echo "Grafana started."

# Give Grafana some time to initialize
sleep 60

echo "Installing kniepdennis-neo4j-datasource plugin..."
# Install the plugin
grafana-cli plugins install kniepdennis-neo4j-datasource
echo "Plugin installed."

echo "Restarting Grafana after plugin installation..."
# Restart Grafana after installing the plugin
killall grafana-server
/run.sh &
echo "Grafana restarted."

# Another wait to ensure Grafana is up after restart
sleep 60

echo "Creating Neo4j_db data source..."
# Use curl to create a new data source
r=$(curl -X POST "http://localhost:3000/api/datasources" \
     -u admin:admin \
     -H "Content-Type: application/json" \
     -d '{
            "name": "Neo4j_db",
            "type": "kniepdennis-neo4j-datasource",
            "url": "neo4j://neo4j:7687",
            "access": "proxy",
            "basicAuth": true,
            "basicAuthUser": "neo4j",
            "secureJsonData": {
                "basicAuthPassword": "password",
                "password": "password"
            },
            "jsonData": {
                "url": "neo4j://neo4j:7687",
                "username": "neo4j"
            },
            "editable": true
         }')

UID_REPLACE_neo=$(echo $r | awk -F, '{for(i=1;i<=NF;i++) {if ($i ~ /"uid":"/) {split($i, a, ":"); print a[2]}}}' | tr -d '"')


echo "Data source created."
echo "r: $r"
echo "Creating dashboard..."
echo "UID_REPLACE: $UID_REPLACE_neo"



DASHBOARD_JSON_CONTENT=$(sed "s/UID_REPLACE_neo/$UID_REPLACE_neo/g" /scripts/dashboard_template.json)

curl -X POST "http://localhost:3000/api/dashboards/db" \
     -u admin:admin \
     -H "Content-Type: application/json" \
     -d "$DASHBOARD_JSON_CONTENT"

echo "Dashboard created."
echo "$DASHBOARD_JSON_CONTENT"
############################prometheus################################
echo "Creating Prometheus data source..."
# Use curl to create a new data source
r=$(curl -X POST "http://localhost:3000/api/datasources" \
     -u admin:admin \
     -H "Content-Type: application/json" \
     -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "editable": true
         }')

UID_REPLACE_prom=$(echo $r | awk -F, '{for(i=1;i<=NF;i++) {if ($i ~ /"uid":"/) {split($i, a, ":"); print a[2]}}}' | tr -d '"')

echo "Prometheus data source created."
echo "r: $r"
echo "Creating Prometheus dashboard..."
echo "UID_REPLACE: $UID_REPLACE_prom"
DASHBOARD_JSON_CONTENT=$(sed "s/UID_REPLACE_prom/$UID_REPLACE_prom/g" /scripts/prometheus_dashboard_template.json)
DASHBOARD_JSON_CONTENT=$(sed "s/UID_REPLACE_neo/$UID_REPLACE_neo/g" <<< "$DASHBOARD_JSON_CONTENT")

curl -X POST "http://localhost:3000/api/dashboards/db" \
     -u admin:admin \
     -H "Content-Type: application/json" \
     -d "$DASHBOARD_JSON_CONTENT"

echo "Prometheus dashboard created."
echo "$DASHBOARD_JSON_CONTENT"

wait

