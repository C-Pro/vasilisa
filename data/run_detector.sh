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
    arecord -D default -r 16000 -f S16_LE -d 2 -t raw tmpfile.raw
    sox -e signed-integer -t raw -b 16 -r 16000 tmpfile.raw -n spectrogram -mr -x 200 -y 99 -o spectrogram.png
    python
done


