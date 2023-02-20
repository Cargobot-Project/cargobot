#!/bin/bash
dir = $(echo %cd%)

docker run --name cargobot_container --privileged -v C:/Users/User/Documents/GitHub/cargobot/:/usr/cargobot -it cargobot_image:latest bash
