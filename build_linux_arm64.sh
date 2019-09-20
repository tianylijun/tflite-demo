#!/bin/bash

mkdir -p build-linux
pushd build-linux
mkdir -p arm64-v8a
pushd arm64-v8a
cmake -DCMAKE_TOOLCHAIN_FILE=./linux.toolchain.cmake ../..
make -j4
popd
popd
cp build-linux/arm64-v8a/tflite-demo /media/psf/Home/nfs/
