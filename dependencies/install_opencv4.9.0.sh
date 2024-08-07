#!/bin/bash -e

BASE_ROOT_DIR=$HOME

install_opencv () {
  cd $BASE_ROOT_DIR
  # Check if the file /proc/device-tree/model exists
  if [ -e "/proc/device-tree/model" ]; then
      # Read the model information from /proc/device-tree/model and remove null bytes
      model=$(tr -d '\0' < /proc/device-tree/model)
      # Check if the model information contains "Jetson Nano Orion"
      echo ""
      if [[ $model == *"Orin"* ]]; then
          echo "Detecting a Jetson Nano Orin."
	  # Use always "-j 4"
          NO_JOB=4
          arch_ver=8.7
          PTX="sm_87"
      elif [[ $model == *"Jetson Nano"* ]]; then
          echo "Detecting a regular Jetson Nano."
          arch_ver=5.3
          PTX="sm_53"
	  # Use "-j 4" only swap space is larger than 5.5GB
	  FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')"
	  if [[ "FREE_MEM" -gt "5500" ]]; then
	    NO_JOB=4
	  else
	    echo "Due to limited swap, make only uses 1 core"
	    NO_JOB=1
	  fi
      else
          echo "Unable to determine the Jetson Nano model."
          exit 1
      fi
      echo ""
  else
      echo "Error: /proc/device-tree/model not found. Are you sure this is a Jetson Nano?"
      exit 1
  fi

  echo "Installing OpenCV 4.9.0 on your Nano"
  echo "It will take 3.5 hours !"

  # reveal the CUDA location
  cd $BASE_ROOT_DIR
  sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
  ldconfig

  # install the Jetson Nano dependencies first
  if [[ $model == *"Jetson Nano"* ]]; then
    apt-get install -y build-essential git unzip pkg-config zlib1g-dev
    apt-get install -y python3-dev python3-numpy
    apt-get install -y python-dev python-numpy
    apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
    apt-get install -y libgstreamer-plugins-good1.0-dev
    apt-get install -y libtbb2 libgtk-3-dev v4l2ucp libxine2-dev
  fi

  if [ -f /etc/os-release ]; then
      # Source the /etc/os-release file to get variables
      . /etc/os-release
      # Extract the major version number from VERSION_ID
      VERSION_MAJOR=$(echo "$VERSION_ID" | cut -d'.' -f1)
      # Check if the extracted major version is 22 or earlier
      if [ "$VERSION_MAJOR" = "22" ]; then
          apt-get install -y libswresample-dev libdc1394-dev
      else
	  apt-get install -y libavresample-dev libdc1394-22-dev
      fi
  else
      apt-get install -y libavresample-dev libdc1394-22-dev
  fi
  # install the common dependencies
  apt-get install -y cmake \
		libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev \
		libpng-dev libtiff-dev libglew-dev \
		libavcodec-dev libavformat-dev libswscale-dev \
		libgtk2.0-dev libgtk-3-dev libcanberra-gtk* \
		python3-pip \
		libxvidcore-dev libx264-dev \
		libtbb-dev libxine2-dev \
		libv4l-dev v4l-utils qv4l2 \
		libtesseract-dev \
		libvorbis-dev \
		libfaac-dev libmp3lame-dev libtheora-dev \
		libopencore-amrnb-dev libopencore-amrwb-dev \
		libopenblas-dev libatlas-base-dev libblas-dev \
		liblapack-dev liblapacke-dev libeigen3-dev gfortran \
		libhdf5-dev libprotobuf-dev protobuf-compiler \
		libgoogle-glog-dev libgflags-dev

  # remove old versions or previous builds
  cd $BASE_ROOT_DIR
  rm -rf opencv*
  # download the 4.9.0 version
  git clone https://github.com/opencv/opencv -b 4.9.0
  git clone https://github.com/opencv/opencv_contrib -b 4.9.0

  # set install dir
  cd $BASE_ROOT_DIR/opencv
  mkdir build
  cd build

  # run cmake
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr \
  -D OPENCV_EXTRA_MODULES_PATH=$BASE_ROOT_DIR/opencv_contrib/modules \
  -D WITH_OPENCL=OFF \
  -D CUDA_ARCH_BIN=${arch_ver} \
  -D CUDA_ARCH_PTX=${PTX} \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_QT=OFF \
  -D WITH_GSTREAMER=ON \
  -D WITH_TBB=ON \
  -D BUILD_TESTS=OFF \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
  -D PYTHON_EXECUTABLE=$(which python3) \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_EXAMPLES=ON \
  -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
  -D CMAKE_C_FLAGS="-march=native -mtune=native" ..

  make -j ${NO_JOB}

  directory="/usr/include/opencv4/opencv2"
  if [ -d "$directory" ]; then
    # Directory exists, so delete it
    rm -rf "$directory"
  fi

  make install
  ldconfig

  make clean
  rm -rf $BASE_ROOT_DIR/opencv
  rm -rf $BASE_ROOT_DIR/opencv_contrib

  echo "You've successfully installed OpenCV 4.9.0 on your Nano"
}

if [ $(python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -q 'NVIDIA CUDA: *NO') ]; then
    echo "Installing OpenCV 4.9.0 with CUDA support"
    install_opencv
else
    echo "OpenCV 4.9.0 with CUDA support is already installed"
fi
