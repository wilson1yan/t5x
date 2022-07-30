#!/bin/sh

gcloud alpha compute tpus tpu-vm create wilson-v3-8-$1 \
    --zone=us-central1-a \
    --accelerator-type='v3-8' \
    --version='tpu-vm-tf-2.9.1' \
