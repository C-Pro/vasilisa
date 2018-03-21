#!/bin/bash
BITRATE=16000
FORMAT=S16_LE
DURATION=2
DEVICE=default
#hw:CARD=U0x46d0x81b,DEV=0

mkdir -p $1
pushd $1

while true
do
    aplay ../laser.wav
    arecord -D $DEVICE -r $BITRATE -f $FORMAT -d $DURATION -t raw --use-strftime rec-%Y%m%dT%H%M%S.raw
done

popd
