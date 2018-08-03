docker run -u user -v ${PWD}:/home/user -w /home/user tiagopeixoto/graph-tool\
    python convert_connectome.py $*
