#!/bin/bash
for f in rec-*.raw
do
    fname=`basename $f`
    sox -e signed-integer -t raw -b 16 -r 16000 $f split%1n-$fname trim 0 2 : newfile : restart
    if [ $? -eq 0 ]
    then
        rm $f
    fi
done


