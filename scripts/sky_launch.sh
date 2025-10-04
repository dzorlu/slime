#!/bin/bash

set -e

CONFIG_FILE="${1:-sky_configs/train.yaml}"
CLUSTER_NAME="${SLIME_CLUSTER_NAME:-slime-training}"

if ! command -v sky &> /dev/null; then
    echo "SkyPilot is not installed. Installing..."
    pip install "skypilot[aws]"
fi

echo "Launching training job with SkyPilot..."
echo "Config: $CONFIG_FILE"
echo "Cluster: $CLUSTER_NAME"

shift
sky launch -c "$CLUSTER_NAME" "$CONFIG_FILE" "$@"
