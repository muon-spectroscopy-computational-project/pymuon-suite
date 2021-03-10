#!/bin/bash

# check if working directory (/home/pymuon-suite) is empty
# if it is empty, container was probably started with `docker run`
# rather than using the Docker Compose file which mounts the folder
if [ "$(ls -A ./pymuonsuite)" ]; then
    #install package in editable mode
    pip install -e .
    #keep container running forever
    tail -f /dev/null
else
    echo "Directory $(pwd)/pymuonsuite is empty."
    echo "The container was probably started without mounting the required folders."
    echo "Make sure you are using the Docker Compose file provided at https://github.com/muon-spectroscopy-computational-project/pymuon-suite/blob/master/docker-compose-dev.yaml"
    echo "Exiting..."
fi
