#!/bin/bash

# Define the target directory
TARGET_DIR="/data/bioasq13/phaseA-reranker/trained_models_b01"

# List only subdirectories
find "$TARGET_DIR" -mindepth 2 -maxdepth 2 -type d