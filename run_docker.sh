xhost +
docker run -it --network=host --gpus all --device /dev/dri --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v `pwd`:`pwd` slam_in_autonomous_driving bash
