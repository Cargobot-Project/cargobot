## __CARGOBOT USER MANUAL__

This a user manual for Cargobot users.
Source code is available: https://github.com/Cargobot-Project/cargobot

### __PRE-REQUISITES__

Docker is strictly required for use of Cargobot, and CUDA availability is recommended. There are different Docker images whether you have CUDA or not.

Also, if you like to reach the GUI of the project, you should install Django on your local. 

Windows and Ubuntu are supported as Operating Systems.

#### __Docker Installation__

Follow the link below and find the best Docker installation specifications for you:
https://docs.docker.com/get-docker/

#### __CUDA Installation__

You have to check that if CUDA can be installed to your computer via https://developer.nvidia.com/cuda-gpus 
Then follow the link below and find the best CUDA installation specifications for you:
https://developer.nvidia.com/cuda-downloads

We tested for CUDA 11.7, but CUDA is not backwards compatible therefore latest version can be downloaded.

### __GET CARGOBOT READY FOR USE__

#### __If CUDA is not available,__ 

1. Make sure that you have downloaded the source code from GitHub link above, and you are in the project directory.
2. Change the file name of NoCUDAAvailable-Dockerfile to Dockerfile.
3. Then build the image as below follows, 
   ``` docker build -t cargobot_image .```
4. Run the image and create container as follows,
   ``` 
   docker run --name cargobot_container \
	--privileged \
	-p 7000:7000 \
   	-p 8888:8888 \
	-v %cd%:/usr/cargobot \
	-it cargobot_image:latest bash 
    ```
5. You should download the segmentation models ".pt" files from the following link: https://drive.google.com/drive/folders/18CqN6_EuQoz0OQr-j_Ue9k9FkD2JsZ7s?usp=sharing
6. After installing the .pt files, migrate those files to under the cargobot/cargobot-project/res/seg
7. Inside the container you can run usr/cargobot/cargobot-project/src/demos, you can find the executable files.
8. After executing any file, the simulation visualization can be reached in http://localhost:7000

#### __If CUDA is available,__

1. Make sure that you have downloaded the source code from GitHub link above, and you are in the project directory.
2. Change the file name of CUDAAvailable-Dockerfile to Dockerfile.
3. Then build the image as below follows, 
   ``` docker build -t cargobot_image .```
4. Run the image and create container as follows,
   ``` 
   docker run --name cargobot_container \
	--runtime=nvidia \
	--gpus all \
	--privileged \
	-p 7000:7000 \
   	-p 8888:8888 \
	-v %cd%:/usr/cargobot \
	-it cargobot_image:latest bash 
    ```
5. You should download the segmentation models ".pt" files from the following link: https://drive.google.com/drive/folders/18CqN6_EuQoz0OQr-j_Ue9k9FkD2JsZ7s?usp=sharing
6. After installing the .pt files, migrate those files to under the cargobot/cargobot-project/res/seg
7. Inside the container you can run usr/cargobot/cargobot-project/src/demos, you can find the executable files.
8. After executing any file, the simulation visualization can be reached in http://localhost:7000

#### How To Reach GUI and give prompts to Cargobot

1. Go to the directory /usr/cargobot/cargobot-project/cargobotgui in your local
2. Run ``` python manage.py runserver ```
3. Then GUI would be available on http://localhost:8000

