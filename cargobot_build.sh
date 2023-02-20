#!/bin/bash
curdir=$(pwd -P)

# Remove any pre-existing container named same with new container
docker rm cargobot_container

docker run --name cargobot_container \
	--privileged \
	-v $curdir:/usr/cargobot \
	-it cargobot_image:latest bash

