#docker run -m 8GB --runtime=nvidia -it --rm  --network host -v ${PWD}:/lab --name multiserver --env TZ=America/Santiago sepulm01/darknet-gpu:ver3 bash exec.sh
sudo docker run --gpus all -it --rm  --network host -v ${PWD}:/lab --name multiserver1 --env TZ=America/Santiago sepulm01/darknet-gpu:ver3 bash exec.sh
