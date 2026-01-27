#!/usr/bin/env bash

find /gadgetron -iname "*.h.yaml" -exec rm {} \; 
find /gadgetron -iname "*.hpp.yaml" -exec rm {} \; 
find /gadgetron -iname "*.hxx.yaml" -exec rm {} \; 
find /gadgetron -iname "*.dp.cpp" -exec rm {} \; 
find /gadgetron -iname "*.dp.hpp" -exec rm {} \; 

git checkout HEAD -- "**/*.hpp"
git checkout HEAD -- "**/*.h"
git checkout HEAD -- "**/*.hxx"

rm -rf /gadgetron/include

