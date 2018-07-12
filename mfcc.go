package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"sync"

	"github.com/remeeting/mrp-go/feat"
)

var (
	canvas [][]color.RGBA
	height = 257
	width  = 100
)

type img struct {
	h, w int
	m    [][]color.RGBA
}

func (m *img) At(x, y int) color.Color { return m.m[x][y] }
func (m *img) ColorModel() color.Model { return color.RGBAModel }
func (m *img) Bounds() image.Rectangle { return image.Rect(0, 0, m.w, m.h) }

func draw(height, width int) {

	fname := "spectrogram.png"
	if len(os.Args) == 2 {
		fname = os.Args[1]
	}

	// open a new file
	f, err := os.Create(fname)
	if err != nil {
		log.Fatal(err)
	}

	img := &img{
		h: height,
		w: width,
		m: canvas,
	}

	fmt.Printf("H: %d, W: %d\n", img.h, img.w)
	// and encode it
	if err := png.Encode(f, img); err != nil {
		log.Fatal(err)
	}
}

func main() {

	// create a canvas
	canvas = make([][]color.RGBA, width)
	for i := range canvas {
		canvas[i] = make([]color.RGBA, height)
	}

	r := bufio.NewReader(os.Stdin)
	var (
		b []int16
		f []float32
	)

	b = make([]int16, 160)
	f = make([]float32, 160)

	var (
		j int = 0
	)

	// Create features
	var (
		zmean   feat.ZeroMean
		preemph feat.Preemphasis
		window  feat.Windowing
		fft     feat.FFT
		mel     feat.MelFilterBank
	)
	// create MFCC pipeline with VTLN
	zmean.In = make(chan []float32)
	zmean.Out = make(chan []float32)
	preemph.In = zmean.Out
	preemph.Out = make(chan []float32)
	preemph.Coeff = 0.97
	window.In = preemph.Out
	window.Out = make(chan []float32)
	window.Type = "hamm"
	fft.In = window.Out
	fft.Out = make(chan []float32)
	fft.Power = true
	mel.In = fft.Out
	mel.Out = make(chan []float32)
	mel.FSize = 10
	mel.Nyquist = 16000.0 / 2.0
	mel.Lo = 188
	mel.Hi = 6071
	mel.NChan = 20
	mel.Vtln = feat.NewVTLN(1.2, mel.Lo, 1500, 5000, mel.Hi, mel.Nyquist)

	run := func(ft feat.Feature) { ft.Compute() }
	go run(zmean)
	go run(preemph)
	go run(window)
	go run(fft)
	go run(mel)

	wg := sync.WaitGroup{}
	wg.Add(1)

	// Receiving goroutine
	go func() {
		maxP := float32(-99999.0)
		minP := float32(99999.0)
		j2 := 0
		for m := range mel.Out {
			for i, p := range m {
				fmt.Printf("p[%d]=%f\n", i, p)
				if p > maxP {
					maxP = p
				}
				if p < minP {
					minP = p
				}
				bright := p*float32(128) + float32(128)
				canvas[j2][i] = color.RGBA{R: uint8(bright), G: uint8(bright), B: uint8(bright), A: 255}
			}
			j2++
		}
		fmt.Printf("Min=%f, max=%f\n", minP, maxP)
		draw(mel.NChan, j2)
		wg.Done()
	}()

	for j < width {
		err := binary.Read(r, binary.LittleEndian, &b)
		if err != nil {
			break
		}
		for i, v := range b {
			f[i] = float32(v) / float32(32767)
			fmt.Printf("f[%d]=%f\n", i, f[i])
		}
		zmean.In <- f
		j++
	}
	close(zmean.In)
	wg.Wait()
}
