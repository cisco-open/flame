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
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

type FileData struct {
	BaseName string `json:"basename"`
	FullName string `json:"fullname"`
	Data     string `json:"data"`
}

func UnzipFile(file *os.File) ([]FileData, error) {
	reader, err := zip.OpenReader(file.Name())
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	fdList := []FileData{}
	for _, f := range reader.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}

		if f.FileInfo().IsDir() {
			// No need to store directory information
			rc.Close()
			continue
		}

		data, err := io.ReadAll(rc)
		if err != nil {
			rc.Close()
			return nil, err
		}

		fd := FileData{
			BaseName: filepath.Base(f.Name),
			FullName: f.Name,
			Data:     string(data),
		}

		fdList = append(fdList, fd)

		rc.Close()
	}

	if len(fdList) == 0 {
		return nil, fmt.Errorf("no file found in %s", file.Name())
	}

	return fdList, nil
}

func ZipFile(fdList []FileData) ([]byte, error) {
	// Create a buffer to write an archive to
	buf := new(bytes.Buffer)

	// Create a new zip archive
	writer := zip.NewWriter(buf)
	for _, fd := range fdList {
		f, err := writer.Create(fd.FullName)
		if err != nil {
			return nil, err
		}

		_, err = f.Write([]byte(fd.Data))
		if err != nil {
			return nil, err
		}
	}

	// Make sure to check the error on Close
	err := writer.Close()
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func ZipFileByTopLevelDir(fdList []FileData) (map[string][]byte, error) {
	fileGroups := make(map[string][]FileData)

	nSplit := 2
	for _, fileData := range fdList {
		splitPaths := strings.SplitN(fileData.FullName, string(os.PathSeparator), nSplit)
		if len(splitPaths) != nSplit {
			return nil, fmt.Errorf("failed to find a top level directory")
		}

		tld := splitPaths[0]
		if _, ok := fileGroups[tld]; !ok {
			fileGroups[tld] = make([]FileData, 0)
		}

		fileGroups[tld] = append(fileGroups[tld], fileData)
	}

	zipFiles := make(map[string][]byte)

	for tld, fdList := range fileGroups {
		zippedData, err := ZipFile(fdList)
		if err != nil {
			return nil, err
		}

		zipFiles[tld] = zippedData
	}

	return zipFiles, nil
}
