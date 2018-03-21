go build -o spectro spectrogram.go && rec  -t raw -b 16 -r 16000 - channels 1 trim 0 00:05 | ./spectro && eog spectrogram.png
