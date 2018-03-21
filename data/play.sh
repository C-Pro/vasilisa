#!/bin/bash
BITRATE=16000
FORMAT=S16_LE
DEVICE=default

pushd $1

for f in *.raw
do
    aplay -D $DEVICE -r $BITRATE -f $FORMAT -t raw $f
done

popd
