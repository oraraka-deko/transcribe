package transcribe

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"

	resampler "github.com/godeps/go-audio-soxr"
)

// QualityPreset defines the resampling quality.
type QualityPreset = resampler.QualityPreset

const (
	QualityQuick    QualityPreset = resampler.QualityQuick
	QualityLow      QualityPreset = resampler.QualityLow
	QualityMedium   QualityPreset = resampler.QualityMedium
	QualityHigh     QualityPreset = resampler.QualityHigh
	QualityVeryHigh QualityPreset = resampler.QualityVeryHigh
)

// TargetSampleRate is the sample rate required by whisper (16kHz).
const TargetSampleRate = 16000

// ResampleMonoFloat32 resamples mono float32 audio data from inputRate to outputRate.
func ResampleMonoFloat32(input []float32, inputRate, outputRate float64, quality QualityPreset) ([]float32, error) {
	if len(input) == 0 {
		return []float32{}, nil
	}
	if inputRate <= 0 || outputRate <= 0 {
		return nil, fmt.Errorf("invalid sample rate: input=%f, output=%f", inputRate, outputRate)
	}
	if inputRate == outputRate {
		cp := make([]float32, len(input))
		copy(cp, input)
		return cp, nil
	}
	return resampler.ResampleMonoFloat32(input, inputRate, outputRate, quality)
}

// ReadWAV reads a WAV stream, converting it to 16kHz mono float32 samples normalized to [-1.0, 1.0].
// If the input audio sample rate is not 16kHz, it is automatically resampled using ResampleMonoFloat32.
// If stereo, channels are downmixed to mono.
func ReadWAV(r io.Reader) ([]float32, error) {
	// Read standard RIFF header (12 bytes)
	riffHeader := make([]byte, 12)
	if _, err := io.ReadFull(r, riffHeader); err != nil {
		return nil, fmt.Errorf("failed to read RIFF header: %w", err)
	}
	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, fmt.Errorf("invalid WAV file: missing RIFF/WAVE signature")
	}

	var channels uint16
	var sampleRate uint32
	var bitsPerSample uint16
	var audioData []byte
	foundFmt := false

	// Iterate through chunks
	chunkHeader := make([]byte, 8)
	for {
		_, err := io.ReadFull(r, chunkHeader)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read chunk header: %w", err)
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		switch chunkID {
		case "fmt ":
			fmtData := make([]byte, chunkSize)
			if _, err := io.ReadFull(r, fmtData); err != nil {
				return nil, fmt.Errorf("failed to read fmt chunk: %w", err)
			}
			if len(fmtData) < 16 {
				return nil, fmt.Errorf("invalid fmt chunk size: %d", len(fmtData))
			}
			audioFormat := binary.LittleEndian.Uint16(fmtData[0:2])
			if audioFormat != 1 { // PCM = 1
				return nil, fmt.Errorf("unsupported audio format %d: only PCM is supported", audioFormat)
			}
			channels = binary.LittleEndian.Uint16(fmtData[2:4])
			sampleRate = binary.LittleEndian.Uint32(fmtData[4:8])
			bitsPerSample = binary.LittleEndian.Uint16(fmtData[14:16])
			foundFmt = true

		case "data":
			if !foundFmt {
				return nil, fmt.Errorf("WAV data chunk found before fmt chunk")
			}
			if chunkSize == 0xFFFFFFFF || chunkSize == 0x7FFFFFFF {
				var err error
				audioData, err = io.ReadAll(r)
				if err != nil {
					return nil, fmt.Errorf("failed to read streaming audio data: %w", err)
				}
			} else {
				audioData = make([]byte, chunkSize)
				if _, err := io.ReadFull(r, audioData); err != nil {
					// In some stream pipes, chunk size in header may slightly exceed available bytes
					if err == io.ErrUnexpectedEOF {
						// read whatever was available
					} else {
						return nil, fmt.Errorf("failed to read audio data: %w", err)
					}
				}
				// Pad byte if chunk size is odd
				if chunkSize%2 != 0 {
					var pad [1]byte
					_, _ = io.ReadFull(r, pad[:])
				}
			}

		default:
			// Skip unknown chunk
			paddedSize := int64(chunkSize)
			if chunkSize%2 != 0 {
				paddedSize++
			}
			if _, err := io.CopyN(io.Discard, r, paddedSize); err != nil && err != io.EOF {
				return nil, fmt.Errorf("failed to skip chunk %s: %w", chunkID, err)
			}
		}

		if len(audioData) > 0 {
			break
		}
	}

	if !foundFmt {
		return nil, fmt.Errorf("WAV format chunk not found")
	}
	if len(audioData) == 0 {
		return nil, fmt.Errorf("no audio data found in WAV")
	}
	if bitsPerSample != 16 {
		return nil, fmt.Errorf("unsupported bit depth: %d (expected 16-bit PCM)", bitsPerSample)
	}
	if channels != 1 && channels != 2 {
		return nil, fmt.Errorf("unsupported channel count: %d (expected 1 for mono or 2 for stereo)", channels)
	}

	var samples []float32
	if channels == 1 {
		numSamples := len(audioData) / 2
		samples = make([]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			rawSample := int16(binary.LittleEndian.Uint16(audioData[i*2 : i*2+2]))
			samples[i] = float32(rawSample) / math.MaxInt16
		}
	} else { // channels == 2, downmix to mono
		numSamples := len(audioData) / 4
		samples = make([]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			left := int16(binary.LittleEndian.Uint16(audioData[i*4 : i*4+2]))
			right := int16(binary.LittleEndian.Uint16(audioData[i*4+2 : i*4+4]))
			samples[i] = (float32(left) + float32(right)) / (2.0 * math.MaxInt16)
		}
	}

	// Resample to 16kHz if necessary
	if sampleRate != TargetSampleRate {
		var err error
		samples, err = ResampleMonoFloat32(samples, float64(sampleRate), float64(TargetSampleRate), QualityHigh)
		if err != nil {
			return nil, fmt.Errorf("failed to resample from %dHz to %dHz: %w", sampleRate, TargetSampleRate, err)
		}
	}

	return samples, nil
}

// ReadWAVFile opens a WAV file from disk and parses it into 16kHz mono float32 samples.
func ReadWAVFile(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open audio file: %w", err)
	}
	defer file.Close()

	return ReadWAV(file)
}

// ReadWAVSamples parses a WAV file into 16kHz float32 samples (alias for ReadWAVFile).
func ReadWAVSamples(path string) ([]float32, error) {
	return ReadWAVFile(path)
}

// ReadAudioFile reads an audio file (WAV, OGG Opus, MP3, AAC, FLAC, M4A, etc.)
// and returns 16kHz mono float32 samples.
// It directly parses WAV files, and falls back to ffmpeg if available for other formats.
func ReadAudioFile(path string) ([]float32, error) {
	// First attempt pure Go WAV parsing
	samples, err := ReadWAVFile(path)
	if err == nil && len(samples) > 0 {
		return samples, nil
	}

	// Try decoding with ffmpeg to 16kHz mono 16-bit PCM WAV
	cmd := exec.Command("ffmpeg", "-y", "-i", path, "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1")
	var out bytes.Buffer
	var errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	cmdErr := cmd.Run()
	if cmdErr == nil && out.Len() > 0 {
		return ReadWAV(&out)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to read audio (wav error: %v, ffmpeg: %v: %s)", err, cmdErr, errBuf.String())
	}
	return nil, fmt.Errorf("no audio samples found in %s", path)
}
