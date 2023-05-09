docker rm drake_torch_container

docker run --name drake_torch_container \
	--runtime=nvidia \
	--gpus all \
	--privileged \
	-p 7000:7000 \
    -p 8888:8888 \
	-v %cd%:/usr/cargobot \
	-it drake_cuda_image:latest bash

