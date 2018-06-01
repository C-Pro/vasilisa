rec -L -t raw -b 16 -r 16000 - channels 1 trim 0 00:05 silence 1 0.1 2% spectrogram -x 256 -y 256 -r -o spectrogram.png  && eog spectrogram.png
