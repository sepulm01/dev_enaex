#docker start cbe68a147fac #levanta docker con mysql
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -d -it --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_1 --env TZ=America/Santiago cam_enaex:v1 
sleep 2
docker run -d -it --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_2 --env TZ=America/Santiago cam_enaex:v1 
sleep 2
docker run -d -it --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_3 --env TZ=America/Santiago cam_enaex:v1 
sleep 2
docker run -d -it --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_4 --env TZ=America/Santiago cam_enaex:v1 
#docker run -d --env="QT_X11_NO_MITSHM=1" --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_1 --env TZ=America/Santiago cam_enaex:latest 
#docker run -d --env="QT_X11_NO_MITSHM=1" --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name cam_2 --env TZ=America/Santiago cam_enaex:latest 
xhost -local:docker




