
for f in  $1/*.raw
do
    sox -e signed-integer -t raw -b 16 -r 16000 $f -n spectrogram -mr -x 200 -y 99 -o $f.png
done
