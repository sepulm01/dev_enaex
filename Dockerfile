FROM python

RUN apt-get -qq update && apt-get -qq install --no-install-recommends -y python3 \ 
 python3-dev \
 python-pil \
 python-lxml \
 python-tk \
 python3-pip \
 build-essential \
 cmake \ 
 git-core \ 
 libgtk2.0-dev \ 
 pkg-config \ 
 libavcodec-dev \ 
 libavformat-dev \ 
 libswscale-dev \ 
 libtbb2 \
 libtbb-dev \ 
 libjpeg-dev \
 libpng-dev \
 libtiff-dev \
 #libjasper-dev \
 libdc1394-22-dev \
 x11-apps \
 wget \
 nano \
 ffmpeg \
 unzip \
 && rm -rf /var/lib/apt/lists/*

#RUN pip3 install --upgrade pip3
RUN pip3 install opencv-contrib-python
RUN pip3 install datetime2
#RUN pip3 install pytest-timeit
#RUN pip3 install pandas
RUN pip3 install Pillow
RUN pip3 install matplotlib
RUN pip3 install descartes
RUN pip3 install Shapely
RUN pip3 install sklearn
RUN pip3 install mysql-connector-python
RUN pip3 install psycopg2-binary
RUN pip3 install cython
RUN pip3 install imagezmq


# Setting up working directory 
RUN mkdir /lab
WORKDIR /lab
#ADD . /lab/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

CMD bash exec.sh