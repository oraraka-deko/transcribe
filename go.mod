module github.com/oraraka-deko/transcribe

go 1.27.1

require (
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0-20260911141324-1da4dc82fa79
	github.com/godeps/go-audio-soxr v0.1.1
)

require (
	github.com/tphakala/simd v1.0.14 // indirect
	golang.org/x/sys v0.38.0 // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
)

replace github.com/ggerganov/whisper.cpp/bindings/go => ./whisper.cpp/bindings/go
