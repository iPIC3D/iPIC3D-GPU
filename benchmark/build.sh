#!/bin/bash


rm -rf build
mkdir build

cd build

cmake -DBENCH_MARK=ON ../../..
make -j

exit $?
