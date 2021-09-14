// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package restapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"text/template"

	"go.uber.org/zap"
)

const (
	// Keys for dataset endpoints
	CreateDatasetEndPoint  = "CREATE_DATASET"
	GetDatasetEndPoint     = "GET_DATASET"
	GetDatasetsEndPoint    = "GET_DATASETS"
	GetAllDatasetsEndPoint = "GET_ALL_DATASETS"

	// Keys for design endpoints
	CreateDesignEndPoint = "CREATE_DESIGN"
	GetDesignsEndPoint   = "GET_DESIGNS"
	GetDesignEndPoint    = "GET_DESIGN"

	// Keys for design schema endpoints
	CreateDesignSchemaEndPoint = "CREATE_DESIGN_SCHEMA"
	GetDesignSchemaEndPoint    = "GET_DESIGN_SCHEMA"
	GetDesignSchemasEndPoint   = "GET_DESIGN_SCHEMAS"
	UpdateDesignSchemaEndPoint = "UPDATE_DESIGN_SCHEMA"

	// Keys for design code endpoints
	CreateDesignCodeEndPoint = "CREATE_DESIGN_CODE"
	GetDesignCodeEndPoint    = "GET_DESIGN_CODE"
	UpdateDesignCodeEndPoint = "UPDATE_DESIGN_CODE"

	// Keys for job endpoints
	CreateJobEndpoint       = "CREATE_JOB"
	SubmitJobEndPoint       = "SUBMIT_JOB"
	GetJobEndPoint          = "GET_JOB"
	GetJobsEndPoint         = "GET_JOBS"
	DeleteJobEndPoint       = "DELETE_JOB"
	UpdateJobEndPoint       = "UPDATE_JOB"
	ChangeJobSchemaEndPoint = "CHANGE_SCHEMA_JOB"

	//Agent
	UpdateAgentStatusEndPoint = "UPDATE_AGENT_STATUS"

	//TODO remove me after prototyping phase is done.
	JobNodesEndPoint = "JOB_NODES"
)

var URI = map[string]string{
	// Dataset
	CreateDatasetEndPoint:  "/{{.user}}/datasets",
	GetDatasetEndPoint:     "/{{.user}}/datasets/{{.datasetId}}",
	GetDatasetsEndPoint:    "/{{.user}}/datasets/?limit={{.limit}}",
	GetAllDatasetsEndPoint: "/datasets",

	// Design
	CreateDesignEndPoint: "/{{.user}}/designs",
	GetDesignEndPoint:    "/{{.user}}/designs/{{.designId}}",
	GetDesignsEndPoint:   "/{{.user}}/designs/?limit={{.limit}}",

	// Design schema
	CreateDesignSchemaEndPoint: "/{{.user}}/designs/{{.designId}}/schemas",
	GetDesignSchemaEndPoint:    "/{{.user}}/designs/{{.designId}}/schemas/{{.version}}",
	GetDesignSchemasEndPoint:   "/{{.user}}/designs/{{.designId}}/schemas",
	UpdateDesignSchemaEndPoint: "/{{.user}}/designs/{{.designId}}/schemas/{{.version}}",

	// Design Code
	CreateDesignCodeEndPoint: "/{{.user}}/designs/{{.designId}}/codes",
	GetDesignCodeEndPoint:    "/{{.user}}/designs/{{.designId}}/codes/{{.version}}",
	UpdateDesignCodeEndPoint: "/{{.user}}/designs/{{.designId}}/codes/{{.version}}",

	// Job
	CreateJobEndpoint:       "/{{.user}}/jobs",
	SubmitJobEndPoint:       "/{{.user}}/jobs",
	GetJobEndPoint:          "/{{.user}}/jobs/{{.jobId}}",
	GetJobsEndPoint:         "/{{.user}}/jobs/?getType={{.type}}&designId={{.designId}}&limit={{.limit}}",
	UpdateJobEndPoint:       "/{{.user}}/jobs/{{.jobId}}",
	DeleteJobEndPoint:       "/{{.user}}/jobs/{{.jobId}}",
	ChangeJobSchemaEndPoint: "/{{.user}}/jobs/{{.jobId}}/schema/{{.schemaId}}/design/{{.designId}}",

	//Agent
	UpdateAgentStatusEndPoint: "/{{.user}}/job/{{.jobId}}/agent/{{.agentId}}",

	//TODO remove me after prototyping phase is done.
	JobNodesEndPoint: "/{{.user}}/nodes",
}

func FromTemplate(skeleton string, inputMap map[string]string) (string, error) {
	//https://stackoverflow.com/questions/29071212/implementing-dynamic-strings-in-golang
	var t = template.Must(template.New("").Parse(skeleton))
	buf := bytes.Buffer{}
	err := t.Execute(&buf, inputMap)
	if err != nil {
		zap.S().Errorf("error creating a text from skeleton. %v", err)
		return "", err
	}
	return buf.String(), nil
}

func CreateURL(ip string, portNo uint16, endPoint string, inputMap map[string]string) string {
	msg, err := FromTemplate(URI[endPoint], inputMap)
	if err != nil {
		zap.S().Errorf("error creating a uri. End point: %s", endPoint)
		return ""
	}
	// TODO: change it to https
	url := "http://" + ip + ":" + strconv.Itoa(int(portNo)) + msg
	return url
}

func HTTPPost(url string, msg interface{}, contentType string) (int, []byte, error) {
	postBody, err := json.Marshal(msg)
	if err != nil {
		zap.S().Errorf("error encoding the payload")
		return -1, nil, err
	}

	resp, err := http.Post(url, contentType, bytes.NewBuffer(postBody))
	if ErrorNilCheck(GetFunctionName(HTTPPost), err) != nil {
		return -1, nil, err
	}
	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if ErrorNilCheck(GetFunctionName(HTTPPost), err) != nil {
		return -1, nil, err
	}

	return resp.StatusCode, body, nil
}

func HTTPPut(url string, msg interface{}, contentType string) (int, []byte, error) {
	putBody, err := json.Marshal(msg)
	if err != nil {
		zap.S().Errorf("error encoding the payload")
		return -1, nil, err
	}

	req, err := http.NewRequest(http.MethodPut, url, bytes.NewBuffer(putBody))
	if ErrorNilCheck(GetFunctionName(HTTPPut), err) != nil {
		return -1, nil, err
	}

	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	client := &http.Client{}
	resp, err := client.Do(req)

	if ErrorNilCheck(GetFunctionName(HTTPPut), err) != nil {
		return -1, nil, err
	}
	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	if ErrorNilCheck(GetFunctionName(HTTPPut), err) != nil {
		return -1, nil, err
	}

	return resp.StatusCode, body, nil
}

func HTTPGet(url string) (int, []byte, error) {
	resp, err := http.Get(url)

	if ErrorNilCheck(GetFunctionName(HTTPGet), err) != nil {
		return -1, nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if ErrorNilCheck(GetFunctionName(HTTPGet), err) != nil {
		return -1, nil, err
	}

	return resp.StatusCode, body, nil
}

func CreateMultipartFormData(kv map[string]io.Reader) (*bytes.Buffer, *multipart.Writer, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	for key, reader := range kv {
		var fw io.Writer
		var err error
		if x, ok := reader.(io.Closer); ok {
			defer x.Close()
		}

		if x, ok := reader.(*os.File); ok {
			// Add a file
			if fw, err = writer.CreateFormFile(key, x.Name()); err != nil {
				return nil, nil, err
			}
		} else {
			// Add other fields
			if fw, err = writer.CreateFormField(key); err != nil {
				return nil, nil, err
			}
		}

		if _, err = io.Copy(fw, reader); err != nil {
			return nil, nil, err
		}
	}
	writer.Close()

	return &buf, writer, nil
}

//ErrorNilCheck logger function to avoid re-writing the checks
func ErrorNilCheck(method string, err error) error {
	if err != nil {
		zap.S().Errorf("[%s] an error occurred %v", method, err)
	}

	return err
}

func GetFunctionName(i interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(i).Pointer()).Name()
}

func CheckStatusCode(code int) error {
	if code >= 400 && code <= 599 {
		return fmt.Errorf("status code: %d", code)
	}

	return nil
}
