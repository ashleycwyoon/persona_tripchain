#!/usr/bin/env bash
# Start the GraphHopper public-transit server used by the environment (localhost:8989).
#
# GraphHopper lives with the data, in $DATA_HOME/tripchain_1030/external/GraphHopper:
# the web jar, a bundled JDK 17, config-pt-rebuilt.yml and gtfs.zip. The config's
# graph.location is the prebuilt graph cache; the OSM extract and gtfs.zip are only
# read when that cache has to be rebuilt. Server output goes to logs/server.out.
set -euo pipefail
cd "${DATA_HOME:-$HOME/data}/tripchain_1030/external/GraphHopper"
mkdir -p logs
exec nohup .jdk17/lib/jvm/bin/java -Xms8g -Xmx32g -jar graphhopper-web-11.0.jar server config-pt-rebuilt.yml > logs/server.out 2>&1
