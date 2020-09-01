FROM ubuntu:18.04

RUN apt-get -qq update && apt-get -qq install --no-install-recommends -y python3 python3-pip 

RUN pip3 install -r requirements.txt

# Setting up working directory 
RUN mkdir /lab
WORKDIR /lab
#ADD . /lab/


# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

CMD bash exec.sh

