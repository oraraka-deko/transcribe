package transcribe

/*
#cgo CFLAGS: -I${SRCDIR}/whisper.cpp/include -I${SRCDIR}/whisper.cpp/ggml/include

// --- Linux Configuration ---
#cgo linux LDFLAGS: -L${SRCDIR}/whisper.cpp/buildlin/src -L${SRCDIR}/whisper.cpp/buildlin/ggml/src -lwhisper -lggml -lstdc++ -lm

// --- Windows Configuration ---
#cgo windows LDFLAGS: -L${SRCDIR}/whisper.cpp/buildwin/src -L${SRCDIR}/whisper.cpp/buildwin/ggml/src -lwhisper -lggml -lstdc++ -lm -lpathcch
*/
import "C"