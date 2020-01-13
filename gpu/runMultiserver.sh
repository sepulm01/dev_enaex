docker run -m 8GB --runtime=nvidia -it --rm  --network host -v ${PWD}:/lab --name multiserver --env TZ=America/Santiago sepulm01/darknet-gpu:ver3
