package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// ReadWAVSamples parses a 16kHz 16-bit Mono WAV file into []float32 samples expected by whisper.cpp
func ReadWAVSamples(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	// Read standard 44-byte WAV header
	header := make([]byte, 44)
	if _, err := io.ReadFull(file, header); err != nil {
		return nil, fmt.Errorf("failed to read WAV header: %w", err)
	}

	// Check for RIFF/WAVE signature
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("file is not a valid RIFF WAV audio file")
	}

	sampleRate := binary.LittleEndian.Uint32(header[24:28])
	channels := binary.LittleEndian.Uint16(header[22:24])
	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])

	if sampleRate != 16000 || channels != 1 || bitsPerSample != 16 {
		return nil, fmt.Errorf("WAV format unsupported: got %dHz %dch %dbit, expected 16000Hz 1ch 16bit", sampleRate, channels, bitsPerSample)
	}

	// Read raw audio payload
	rawData, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio data: %w", err)
	}

	// Convert 16-bit PCM (int16) to float32 normalized [-1.0, 1.0]
	numSamples := len(rawData) / 2
	samples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		rawSample := int16(binary.LittleEndian.Uint16(rawData[i*2 : i*2+2]))
		samples[i] = float32(rawSample) / math.MaxInt16
	}

	return samples, nil
}

func main() {
	modelPath := "ggml-base-q8_0.bin"
	audioPath := "audio.wav"

	// 1. Extract PCM samples
	samples, err := ReadWAVSamples(audioPath)
	if err != nil {
		log.Fatalf("Audio Error: %v", err)
	}

	// 2. Load Whisper model
	model, err := whisper.New(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()

	// 3. Init execution context
	context, err := model.NewContext()
	if err != nil {
		log.Fatalf("Failed to create context: %v", err)
	}

	// Set processing parameters
	if err := context.SetLanguage("en"); err != nil {
		log.Printf("Warning: failed to set language: %v", err)
	}
	context.SetThreads(4)

	// 4. Process audio buffer (requires 4 arguments: samples, cbEncoderBegin, cbSegment, cbProgress)
	log.Println("Transcribing audio...")
	if err := context.Process(samples, nil, nil, nil); err != nil {
		log.Fatalf("Transcription error: %v", err)
	}

	// 5. Retrieve output segments
	fmt.Println("\n--- Transcript ---")
	for {
		segment, err := context.NextSegment()
		if err != nil {
			break
		}
		fmt.Printf("[%s -> %s] %s\n", segment.Start, segment.End, segment.Text)
	}
}