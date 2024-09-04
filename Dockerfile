
# To use image, you need to install Nvidia Container toolkit on your host machine and configure
# Please refer to: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Pass a cuda toolkit version to fit your host driver, 12.4.0 requires a 550
ARG CUDA_VERSION=12.4.0 
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    --no-install-recommends \
    build-essential \
    cmake \
    wget \
    libhdf5-dev \
    mpich \
    libmpich-dev \
    zip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/iPIC3D-CUDA
COPY . .

# Build iPIC3D-CUDA
RUN mkdir -p build && \
    cd build && \
    rm -rf ./* && \
    cmake .. && \
    make -j

# Execute iPIC3D-CUDA

CMD ["bash"]
