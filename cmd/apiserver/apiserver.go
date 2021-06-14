package main

import (
	"os"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/cmd"
)

/*
	https://pkg.go.dev/go.uber.org/zap#ReplaceGlobals
	https://pkg.go.dev/go.uber.org/zap#hdr-Choosing_a_Logger

	In most circumstances, use the SugaredLogger. It's 4-10x faster than most
	other structured logging packages and has a familiar, loosely-typed API.
		sugar := logger.Sugar()
		sugar.Infow("Failed to fetch URL.",
			// Structured context as loosely typed key-value pairs.
			"url", url,
			"attempt", 3,
			"backoff", time.Second,
		)
		sugar.Infof("Failed to fetch URL: %s", url)

	In the unusual situations where every microsecond matters, use the
	Logger. It's even faster than the SugaredLogger, but only supports
	structured logging.
		logger.Info("Failed to fetch URL.",
			// Structured context as strongly typed fields.
			zap.String("url", url),
			zap.Int("attempt", 3),
			zap.Duration("backoff", time.Second),
		)
*/
func initZapLog() *zap.Logger {
	config := zap.NewDevelopmentConfig()
	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	logger, _ := config.Build()
	return logger
}

func main() {
	loggerMgr := initZapLog() //use zap.S().Infof("") or zap.L().Infof("")
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync() // flushes buffer, if any

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
