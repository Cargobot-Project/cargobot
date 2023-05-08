#!/bin/bash
curdir=$(pwd -P)

# Remove any pre-existing container named same with new container
docker rm drake_torch_container

docker run --name drake_torch_container \
	--runtime=nvidia \
	--gpus all \
	--privileged \
	-p 7000:7000 \
    -p 8888:8888 \
	-v $curdir:/usr/cargobot \
	-it drake_cuda_image:latest bash

