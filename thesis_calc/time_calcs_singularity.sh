#!/bin/bash

timestamp() {
  date +%s
}

STARTING=$(timestamp)
./glaunet_singularity.sh
ENDING=$(timestamp)
echo "=========================="
echo $STARTING
echo $ENDING
expr $ENDING - $STARTING
