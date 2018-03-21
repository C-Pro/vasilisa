CGO_CPPFLAGS="-static -Bstatic" GOARCH=arm GOARM=5 CGO_ENABLED=1 CC=/usr/bin/arm-linux-gnueabi-gcc go build -o main main.go
