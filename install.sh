#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEPENDENCIES_DIR=$SCRIPT_DIR/dependencies

install_opencv() {
    $DEPENDENCIES_DIR/install_opencv4.9.0.sh
}

install_dependencies() {
    pip install -r $DEPENDENCIES_DIR/requirements.txt
    install_opencv
}

install_dependencies