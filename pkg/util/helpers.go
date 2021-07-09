package util

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// InitZapLog Zap Logger initialization
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
func InitZapLog(service string) *zap.Logger {
	dirPath := filepath.Join("/var/log", ProjectName)
	logPath := filepath.Join(dirPath, service+".log")
	err := os.MkdirAll(dirPath, 0755)
	if err != nil {
		fmt.Printf("Can't create directory: %v\n", err)
		return nil
	}

	config := zap.NewDevelopmentConfig()
	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	config.OutputPaths = []string{
		"stdout",
		logPath,
	}

	logger, err := config.Build()
	if err != nil {
		fmt.Printf("Can't build logger: %v", err)
		return nil
	}

	return logger
}

//ErrorNilCheck logger function to avoid re-writing the checks
func ErrorNilCheck(method string, err error) {
	if err != nil {
		zap.S().Errorf("[%s] an error occurred %v", method, err)
	}
}

func GetFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

//FormatJSON prettify the json by adding tabs
func FormatJSON(data []byte) ([]byte, error) {
	var out bytes.Buffer
	err := json.Indent(&out, data, "", "    ")
	if err == nil {
		return out.Bytes(), err
	}
	return data, nil
}
