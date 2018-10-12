#!/usr/local/bin/bash

docker run -p 8888:8888 -p 6006:6006 -v ${PWD}:/home/user -it -u user -w /home/user $1 bash
