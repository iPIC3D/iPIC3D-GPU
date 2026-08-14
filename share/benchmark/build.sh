#!/bin/bash


rm -rf build
mkdir build

cd build

cmake -DBENCHMARK_MODE=2 -DTIME_TASKS=ON ../../..
make -j

exit $?
