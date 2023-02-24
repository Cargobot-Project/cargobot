# Cargobot

## Building the Environment in Docker

First, make sure that "Docker" is fully installed to your machine.

Then you have to build the cargobot image based on the robotlocomotion/drake:focal image. You have to be in the directory consisting Dockerfile file. After, you can run the command below.

```
docker build -t cargobot_image .
```

Now, our docker image is built with dependencies we want. To run the image and create a container, you have to run the commands below according to your operating system.

Linux:

```
sh cargobot_build.sh
```

Windows:

```
cargobot_build.bat
```

Now, you are in the bash of the cargobot_container that we built. Virtual environment is already created but cannot be run outside of the container. So you have to activate it with command below.

```
cd /
. venv/bin/activate
```

Now you would be able to compile and run any .py file you want.

## Notes - Please Read

### Version Control

If you have pulled anything from github you must create the container again, but don't have to create the image again. Therefore, you should run cargobot_build.sh or cargobot_build.bat according to your operating system as above.

#### Removing Docker Images and Containers

Name of the image created according to Dockerfile is "cargobot_image". To remove it, you should remove all the containers binded to this image. Therefore, see all containers with command below.

Name of the container created according to scripts are "cargobot_container". Shell scripts also removes any container named "cargobot_container" to prevent any errors you may have running the scripts. You can see how to delete any container manually from below.

#### Delete Running/Stopped Container

Learn the running containers with command below.
```
docker ps
```
Fetch the name of the container you wish to be stopped and removed (e.g. container name is ugly_panda).

```
docker stop ugly_panda

docker rm ugly_panda
```

Also you can stop the container and start it again with: 

```
docker start ugly_panda
```

Learn the stopped containers with command below.

```
docker ps -a
```

Then you can remove them as is above.


### How to See Meshcat Visualizer

First of all, the "cargobot_container"s port 7000 is exposed to the localhost:7000. So Meschat Visualizer can be seen in 

```
localhost:7000
```

#### Windows 10

In our tests on W10, Meshcat Visualizer can be seen on Firefox but not Chrome. Therefore, make sure that Firefox is downloaded on your Windows local host. After you run code including visualizer. You can go to the link "localhost:7000" at Firefox and access the visualizer.

#### Ubuntu 20.04

In our tests on Ubuntu 20.04, Meshcat Visualizer can be seen on Chrome but not Firefox. Therefore, make sure that Chrome is downloaded on your Ubuntu local host. After you run code including visualizer. You can go to the link "localhost:7000" at Chrome and access the visualizer.

## Manipulating Code in Docker Container from VSCode

#TODO

## Building the Environment to Your Local

#TODO
