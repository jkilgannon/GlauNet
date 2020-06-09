#!/bin/bash

filewatched="/outgoing/best_model_incremental-cls1-run$1"
fileext='h5'
serialnum=1

outputdata=$(stat -c '%y' "$filewatched.$fileext")
outputdataprev=$outputdata

function act_when_modified {
   echo 'File was modified. Starting processing (includes 45 second sleep).'

   # Wait until the file has been fully written.
   sleep 45

   # Run the process that turns the saved model into human-readable data
   serialnumstr="$serialnum"
   ./check_model.sh $1 $serialnumstr

   echo "Processing $serialnum complete."
   serialnum=$((serialnum+1))
}

while true
do
   outputdataprev=$(stat -c '%y' "$filewatched.$fileext")
   if [ "$outputdata" != "$outputdataprev" ]; then
       act_when_modified $1
   fi
   outputdata=$outputdataprev
   sleep 10
done
