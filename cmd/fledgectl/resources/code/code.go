package code

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
)

type Params struct {
	resources.CommonParams

	DesignId string
	CodePath string
	CodeVer  string
}

func Create(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.CreateDesignCodeEndPoint, uriMap)

	// "fileName", "fileVer" and "fileData" are names of variables used in openapi specification
	kv := map[string]io.Reader{
		"fileName": strings.NewReader(filepath.Base(params.CodePath)),
		"fileVer":  strings.NewReader(params.CodeVer),
		"fileData": mustOpen(params.CodePath),
	}

	// create multipart/form-data
	buf, writer, err := restapi.CreateMultipartFormData(kv)
	if err != nil {
		fmt.Printf("Failed to create multipart/form-data: %v\n", err)
		return nil
	}

	// send post request
	resp, err := http.Post(url, writer.FormDataContentType(), buf)
	if err != nil {
		fmt.Printf("Failed to create a code: %v\n", err)
		return nil
	}
	defer resp.Body.Close()

	if err != nil || restapi.CheckStatusCode(resp.StatusCode) != nil {
		fmt.Printf("Failed to create a code - code: %d, error: %v\n", resp.StatusCode, err)
		return nil
	}

	fmt.Printf("Code created successfully for design %s\n", params.DesignId)

	return nil
}

func mustOpen(f string) *os.File {
	r, err := os.Open(f)
	if err != nil {
		zap.S().Fatalf("Failed to open %s: %v", f, err)
	}

	return r
}
