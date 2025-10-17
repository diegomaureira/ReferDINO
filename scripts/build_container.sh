#!/bin/bash

# Name of the Docker image
IMAGE_NAME="referdino"

# Build the Docker image
docker build -t $IMAGE_NAME .