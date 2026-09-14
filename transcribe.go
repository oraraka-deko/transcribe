package transcribe

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// Config configures the Transcriber instance.
type Config struct {
	// ModelPath is the file path to the Whisper ggml model file (e.g. ggml-tiny-q8_0.bin).
	ModelPath string

	// Language specifies the target audio language (e.g., "en", "es", "auto").
	// Default is "auto" (or "en" if model is multilingual and none specified).
	Language string

	// Threads is the number of CPU threads to use for inference.
	// Defaults to min(4, runtime.NumCPU()).
	Threads int

	// Translate controls whether to translate speech to English.
	Translate bool
}

// Segment represents a transcribed text segment with timing information.
type Segment struct {
	Num   int           `json:"num"`
	Start time.Duration `json:"start"`
	End   time.Duration `json:"end"`
	Text  string        `json:"text"`
}

// Transcriber provides high-performance speech-to-text transcription using whisper.cpp.
type Transcriber struct {
	mu      sync.Mutex
	model   whisper.Model
	context whisper.Context
	config  Config
}

// New creates and initializes a new Transcriber with the given configuration.
func New(cfg Config) (*Transcriber, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("model path cannot be empty")
	}

	model, err := whisper.New(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load whisper model %q: %w", cfg.ModelPath, err)
	}

	ctx, err := model.NewContext()
	if err != nil {
		_ = model.Close()
		return nil, fmt.Errorf("failed to create whisper context: %w", err)
	}

	threads := cfg.Threads
	if threads <= 0 {
		threads = runtime.NumCPU()
		if threads > 4 {
			threads = 4
		}
	}
	ctx.SetThreads(uint(threads))

	if cfg.Language != "" {
		if err := ctx.SetLanguage(cfg.Language); err != nil {
			// If not multilingual, log or ignore, but if user explicitly provided a language report error
			if cfg.Language != "auto" && cfg.Language != "en" {
				_ = model.Close()
				return nil, fmt.Errorf("failed to set language %q: %w", cfg.Language, err)
			}
		}
	}

	if cfg.Translate {
		ctx.SetTranslate(true)
	}

	return &Transcriber{
		model:   model,
		context: ctx,
		config:  cfg,
	}, nil
}

// NewSimple creates a Transcriber with default settings for a given model path.
func NewSimple(modelPath string) (*Transcriber, error) {
	return New(Config{
		ModelPath: modelPath,
		Language:  "auto",
	})
}

// Close releases the underlying whisper model and context resources.
func (t *Transcriber) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.model != nil {
		err := t.model.Close()
		t.model = nil
		t.context = nil
		return err
	}
	return nil
}

// TranscribeSegments transcribes 16kHz float32 audio samples and returns all individual timed segments.
func (t *Transcriber) TranscribeSegments(samples []float32) ([]Segment, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.context == nil {
		return nil, fmt.Errorf("transcriber is closed")
	}

	if err := t.context.Process(samples, nil, nil, nil); err != nil {
		return nil, fmt.Errorf("whisper process error: %w", err)
	}

	var segments []Segment
	for {
		seg, err := t.context.NextSegment()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("error reading segment: %w", err)
		}
		segments = append(segments, Segment{
			Num:   seg.Num,
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Text,
		})
	}

	return segments, nil
}

// Transcribe transcribes 16kHz float32 audio samples and returns the concatenated transcribed text.
func (t *Transcriber) Transcribe(samples []float32) (string, error) {
	segments, err := t.TranscribeSegments(samples)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	for i, seg := range segments {
		if i > 0 && !strings.HasSuffix(sb.String(), " ") {
			sb.WriteString(" ")
		}
		sb.WriteString(strings.TrimSpace(seg.Text))
	}
	return sb.String(), nil
}

// TranscribeAudio transcribes audio with any given sample rate, automatically resampling to 16kHz if needed.
func (t *Transcriber) TranscribeAudio(samples []float32, sampleRate int) (string, error) {
	if sampleRate != TargetSampleRate {
		var err error
		samples, err = ResampleMonoFloat32(samples, float64(sampleRate), float64(TargetSampleRate), QualityHigh)
		if err != nil {
			return "", fmt.Errorf("resampling audio failed: %w", err)
		}
	}
	return t.Transcribe(samples)
}

// TranscribeFile parses a WAV file (any sample rate, mono or stereo) and transcribes its audio.
func (t *Transcriber) TranscribeFile(audioPath string) (string, error) {
	samples, err := ReadWAVFile(audioPath)
	if err != nil {
		return "", fmt.Errorf("failed to read WAV file: %w", err)
	}
	return t.Transcribe(samples)
}

// TranscribeFileSegments parses a WAV file and returns all transcribed segments with timing.
func (t *Transcriber) TranscribeFileSegments(audioPath string) ([]Segment, error) {
	samples, err := ReadWAVFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read WAV file: %w", err)
	}
	return t.TranscribeSegments(samples)
}

// QuickTranscribe is a package-level helper to transcribe an audio file in one step.
func QuickTranscribe(modelPath, audioPath string) (string, error) {
	tr, err := NewSimple(modelPath)
	if err != nil {
		return "", err
	}
	defer tr.Close()

	return tr.TranscribeFile(audioPath)
}
