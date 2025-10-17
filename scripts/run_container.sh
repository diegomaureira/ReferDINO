#!/bin/bash

# Host directories
MEDIA_DIR="/media/disk/ReferDINO/"


# Run Docker container with volume mounts and GPU support
docker run --gpus all -it \
  -v "$MEDIA_DIR":/ReferDINO/ \
  --network host \
  referdino

# chmod +x run_container.sh
