FROM ubuntu:26.04

ARG CMAKE_VERSION=3.31.8
ARG CMAKE_SHA256=630615d8e98ac33eba7fbe472626dff5c899c85af3c024585ae109166a6909d0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc-12 g++-12 binutils ca-certificates curl file git gnupg \
    nasm ninja-build pkg-config python3 xz-utils \
    libgtk-3-dev libasound2-dev libpulse-dev libudev-dev libdbus-1-dev libibus-1.0-dev \
    libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev \
    libxss-dev libxtst-dev libxinerama-dev libxkbcommon-dev \
    libwayland-dev wayland-protocols libdecor-0-dev \
    libdrm-dev libgbm-dev \
    libboost-dev \
    squashfs-tools zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 3 -o /tmp/cmake.tar.gz \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    && echo "${CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/cmake \
    && tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1 \
    && rm /tmp/cmake.tar.gz

ENV CARGO_HOME=/opt/rust/cargo \
    RUSTUP_HOME=/opt/rust/rustup \
    PATH=/opt/rust/cargo/bin:/opt/cmake/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CC=gcc-12 \
    CXX=g++-12 \
    CMAKE=/opt/cmake/bin/cmake

WORKDIR /src
