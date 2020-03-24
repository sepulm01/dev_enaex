#docker start cbe68a147fac #levanta docker con mysql
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -m 12GB --runtime=nvidia  -d --env="QT_X11_NO_MITSHM=1" --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -p 12347:12347 -p 5000:5000 -p 8888:8888 -v ${PWD}:/lab --name cam_6 --env TZ=America/Santiago cam_enaex:latest 
xhost -local:docker




