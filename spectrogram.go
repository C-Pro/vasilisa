package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/mjibson/go-dsp/spectral"
)

const (
	minFreq = 200
	maxFreq = 5000
)

var (
	canvas [][]color.RGBA
	height = 257
	width  = 1024
)

type img struct {
	h, w int
	m    [][]color.RGBA
}

func (m *img) At(x, y int) color.Color { return m.m[x][y] }
func (m *img) ColorModel() color.Model { return color.RGBAModel }
func (m *img) Bounds() image.Rectangle { return image.Rect(0, 0, m.w, m.h) }

func compress(x float64) uint8 {
	//1/((1/255)+e^-log(x))
	return uint8(float64(1) / ((float64(1) / float64(255)) + math.Exp(-(math.Log(x) / math.Log(16)))))
}

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
		b    []int16
		f    []float64
		opts spectral.PwelchOptions
	)
	opts.NFFT = 512
	opts.Noverlap = 128

	b = make([]int16, 160)
	f = make([]float64, 160)

	var (
		minI int = 0
		maxI int = height
		j    int = 0
	)

	for j < width {
		err := binary.Read(r, binary.LittleEndian, &b)
		if err != nil {
			draw(height, j)
			return
		}
		//maxVal := -math.MaxFloat64
		for i, v := range b {
			f[i] = float64(v) / 32767
			//	if maxVal < math.Abs(float64(v)) {
			//		maxVal = math.Abs(float64(v))
			//	}
		}
		// Normalize values to 65536
		//for i, _ := range f {
		//	if f[i] != float64(0) {
		//		f[i] = (f[i] / maxVal) * float64(65536)
		//	}
		//}

		amp, freq := spectral.Pwelch(f, 16000, &opts)
		for i, _ := range amp {
			fmt.Printf("%d\t: %d\n", int(freq[i]), compress(freq[i]))
			if i > minI && freq[i] < minFreq {
				minI = i
			}
			if i < maxI && freq[i] > maxFreq {
				maxI = i
			}
		}
		height = maxI - minI - 1

		for i, v := range amp {
			if freq[i] >= minFreq && freq[i] <= maxFreq {
				if v < 0 {
					fmt.Println(v)
				}
				p := compress(v)
				canvas[j][i-minI-1] = color.RGBA{R: p, G: p, B: p, A: 255}
			}
		}
		j++
	}
	draw(height, j)
}
