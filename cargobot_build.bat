:: Remove any pre-existing container named same with new container
docker rm cargobot_container

docker run --name cargobot_container --privileged -v %cd%:/usr/cargobot -p 7000:7000 -it yagizyasarr/cargobot:1.0 bash
    

