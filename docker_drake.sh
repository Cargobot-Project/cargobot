#!/bin/bash

docker run --name cargobot_container \
	--privileged \
	-v ~/CS491/cargobot:/usr/cargobot \
	-it cargobot_image:latest bash

