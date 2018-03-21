package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

func main() {
	r := bufio.NewReader(os.Stdout)
	var b [16]int16
	for {
		if err := binary.Read(b, binary.LittleEndian, &b); err != nil {
			break
		}
		fmt.Printf("Bytes: %+v\n", b)
	}
	fmt.Println("EOF")
	return
}
