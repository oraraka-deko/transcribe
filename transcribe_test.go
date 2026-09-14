package transcribe_test

import (
	"math"
	"os"
	"testing"
	"time"

	"transcribe"
)

func TestResampleMonoFloat32(t *testing.T) {
	// Generate 1 second of 440Hz sine wave at 44100Hz
	inRate := 44100.0
	outRate := 16000.0
	numSamples := int(inRate)
	input := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		input[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / inRate))
	}

	resampled, err := transcribe.ResampleMonoFloat32(input, inRate, outRate, transcribe.QualityHigh)
	if err != nil {
		t.Fatalf("ResampleMonoFloat32 error: %v", err)
	}

	// Expected sample count should be approximately outRate (16000)
	expectedSamples := int(outRate)
	diff := math.Abs(float64(len(resampled) - expectedSamples))
	if diff > 100 {
		t.Fatalf("Expected ~%d samples, got %d", expectedSamples, len(resampled))
	}
	t.Logf("Resampled %d samples (44.1kHz) -> %d samples (16kHz)", len(input), len(resampled))
}

func TestReadWAV(t *testing.T) {
	audioPath := "audio.wav"
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skip("audio.wav not found, skipping test")
	}

	samples, err := transcribe.ReadWAVFile(audioPath)
	if err != nil {
		t.Fatalf("ReadWAVFile error: %v", err)
	}

	if len(samples) == 0 {
		t.Fatal("Expected non-empty audio samples")
	}

	// Verify sample range
	for i, s := range samples {
		if s < -1.5 || s > 1.5 {
			t.Fatalf("Sample at %d out of bounds: %f", i, s)
		}
	}
	t.Logf("Successfully read %d samples from %s", len(samples), audioPath)
}

func TestTranscriber(t *testing.T) {
	modelPath := "ggml-tiny-q8_0.bin"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = "ggml-base-q8_0.bin"
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("No whisper model found, skipping transcription test")
	}

	audioPath := "audio.wav"
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skip("audio.wav not found, skipping transcription test")
	}

	tr, err := transcribe.New(transcribe.Config{
		ModelPath: modelPath,
		Language:  "en",
		Threads:   4,
	})
	if err != nil {
		t.Fatalf("Failed to initialize Transcriber: %v", err)
	}
	defer tr.Close()

	start := time.Now()
	text, err := tr.TranscribeFile(audioPath)
	if err != nil {
		t.Fatalf("TranscribeFile failed: %v", err)
	}
	elapsed := time.Since(start)

	if len(text) == 0 {
		t.Fatal("Transcribed text is empty")
	}

	t.Logf("Transcribed in %v: %q", elapsed, text)
}
