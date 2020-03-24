cd gpu
docker run -d --gpus all -it --rm  --network host -v ${PWD}/gpu:/lab --name multiserver --env TZ=America/Santiago sepulm01/darknet-gpu:ver3 bash exec.sh
cd ..
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth-n
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
docker run -d --env="QT_X11_NO_MITSHM=1" --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name wk_1 --env TZ=America/Santiago cam_enaex:v2 
sleep 2
docker run -d --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name wk_2 --env TZ=America/Santiago cam_enaex:v2 
sleep 2
docker run -d --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name wk_3 --env TZ=America/Santiago cam_enaex:v2 
sleep 2
docker run -d --env="QT_X11_NO_MITSHM=1"  --rm --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --network host -v ${PWD}:/lab --name wk_4 --env TZ=America/Santiago cam_enaex:v2  
xhost -local:docker




