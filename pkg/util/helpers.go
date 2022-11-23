// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package util

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	runes = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

/*
InitZapLog Zap Logger initialization

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
	logPath := filepath.Join(LogDirPath, service+".log")
	err := os.MkdirAll(LogDirPath, FilePerm0755)
	if err != nil {
		fmt.Printf("Can't create directory: %v\n", err)
		return nil
	}

	config := zap.NewDevelopmentConfig()
	//default
	//config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	//config.EncoderConfig.TimeKey = "timestamp"
	//config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	//custom
	config.EncoderConfig.EncodeLevel = func(level zapcore.Level, enc zapcore.PrimitiveArrayEncoder) {
		enc.AppendString("[" + level.CapitalString() + "]")
	}
	config.EncoderConfig.EncodeTime = func(t time.Time, enc zapcore.PrimitiveArrayEncoder) {
		enc.AppendString(t.Format("2006/01/02 15:04:05"))
	}

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

// FormatJSON prettify the json by adding tabs
func FormatJSON(data []byte) ([]byte, error) {
	var out bytes.Buffer
	err := json.Indent(&out, data, "", "    ")
	if err == nil {
		return out.Bytes(), err
	}
	return data, nil
}

// Marshal JSON to escape HTML characters like <, > while printing
func JSONMarshal(t interface{}) ([]byte, error) {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(t)
	return buffer.Bytes(), err
}

// StructToMapInterface converts any struct interface into map interface using json de-coding/encoding.
func StructToMapInterface(in interface{}) (map[string]interface{}, error) {
	//todo maybe a better way to do this https://github.com/golang/protobuf/issues/1259#issuecomment-750453617
	var out map[string]interface{}
	inMarsh, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(inMarsh, &out)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func ToProtoStruct(in interface{}) (*structpb.Struct, error) {
	m, err := StructToMapInterface(in)
	if err != nil {
		zap.S().Errorf("error converting notification object into map interface. %v", err)
		return nil, err
	}
	details, err := structpb.NewStruct(m)
	if err != nil {
		zap.S().Errorf("error creating proto struct. %v", err)
		return nil, err
	}
	return details, nil
}

func ProtoStructToStruct(msg *structpb.Struct, obj interface{}) error {
	inJson, err := msg.MarshalJSON()
	if err != nil {
		return err
	}
	err = json.Unmarshal(inJson, obj)
	if err != nil {
		//zap.S().Errorf("error while converting decoded message into struct. %v", err)
		return err
	}
	return nil
}

func ByteToStruct(msg []byte, obj interface{}) error {
	err := json.Unmarshal(msg, obj)
	if err != nil {
		return err
	}
	return nil
}

func CopyFile(src string, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_EXCL, FilePerm0644)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}

	return out.Close()
}

func RandString(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = runes[rand.Intn(len(runes))]
	}

	return string(b)
}

func Contains(haystack []string, needle string) bool {
	for i := range haystack {
		if needle == haystack[i] {
			return true
		}
	}

	return false
}

func PrettyJsonString(data []byte) (string, error) {
	var prettyJSON bytes.Buffer

	prefix := ""
	indent := "    "
	if err := json.Indent(&prettyJSON, data, prefix, indent); err != nil {
		return "", err
	}

	return prettyJSON.String(), nil
}

func ReadFileToStruct(filePath string, obj interface{}) error {
	content, err := os.ReadFile(filePath)
	if err != nil {
		zap.S().Errorf("error when opening file %s %v", filePath, err)
		return err
	}

	// Now let's unmarshall the data into `payload`
	return ByteToStruct(content, obj)
}
