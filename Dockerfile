FROM ros:noetic

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

ARG user_name
ARG user_id
ARG group_id
ARG workdir
ARG homedir

# Install prerequisite tools and libraries
RUN apt-get update && \
    apt-get install -y sudo wget curl unzip vim git cmake gdb htop libeigen3-dev libgl1-mesa-dev libglew-dev qtbase5-dev libgoogle-glog-dev libpcl-dev libopencv-dev ros-noetic-pcl-ros libyaml-cpp-dev ros-noetic-velodyne-msgs libsuitesparse-dev libgtest-dev ros-noetic-angles pcl-tools libbtbb-dev

# Install cuda toolkit
RUN apt-get install -y nvidia-cuda-toolkit

# Install pangolin recommended prerequisites 
RUN apt-get install -y libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev libc++-dev g++ ninja-build libjpeg-dev libpng-dev libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev

# Install pangolin
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install

ENV DISPLAY=:1

RUN mkdir -p ${workdir}
WORKDIR ${workdir}

RUN touch ${homedir}/.bashrc
RUN echo "alias ll='ls -alF --color=auto'" > ${homedir}/.bashrc

RUN groupadd -g ${group_id} ${user_name} && \
    useradd -u ${user_id} -g ${group_id} -ms /bin/bash ${user_name}
USER ${user_name}

