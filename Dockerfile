FROM robotlocomotion/drake:focal

LABEL maintainer="yagiz@cargobot"

RUN apt-get update && apt-get install -y vim && apt-get install -y python3.8-venv


RUN pip install virtualenv \
	&& virtualenv venv 
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH
RUN which python

RUN /venv/bin/pip install manipulation \
	&& /venv/bin/pip install scipy

ENTRYPOINT /bin/bash


