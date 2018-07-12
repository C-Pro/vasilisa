go build -o spec2 mfcc.go && \
rec  -t raw -b 16 -r 16000 - channels 1 trim 0 00:05 | ./spec2
