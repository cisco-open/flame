package restapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"runtime"
	"strconv"
	"text/template"

	"go.uber.org/zap"
)

//URI must always end with a backslash. Make sure the query params are not ended with a backslash
//API END POINTS
const (
	//Design
	CreateDesignEndPoint       = "CREATE_DESIGN"
	GetDesignsEndPoint         = "GET_DESIGNS"
	GetDesignEndPoint          = "GET_DESIGN"
	CreateDesignSchemaEndPoint = "CREATE_DESIGN_SCHEMA"
	GetDesignSchemaEndPoint    = "GET_DESIGN_SCHEMA"
	GetDesignSchemasEndPoint   = "GET_DESIGN_SCHEMAS"
	UpdateDesignSchemaEndPoint = "UPDATE_DESIGN_SCHEMA"

	//Job
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
	// Design Template
	CreateDesignEndPoint:       "/{{.user}}/designs",
	GetDesignEndPoint:          "/{{.user}}/designs/{{.designId}}",
	GetDesignsEndPoint:         "/{{.user}}/designs/?limit={{.limit}}",
	CreateDesignSchemaEndPoint: "/{{.user}}/designs/{{.designId}}/schemas",
	GetDesignSchemaEndPoint:    "/{{.user}}/designs/{{.designId}}/schemas/{{.version}}",
	GetDesignSchemasEndPoint:   "/{{.user}}/designs/{{.designId}}/schemas",
	UpdateDesignSchemaEndPoint: "/{{.user}}/designs/{{.designId}}/schemas/{{.version}}",

	//Job
	SubmitJobEndPoint:       "/{{.user}}/job",
	GetJobEndPoint:          "/{{.user}}/job/{{.jobId}}",
	GetJobsEndPoint:         "/{{.user}}/jobs/?getType={{.type}}&designId={{.designId}}&limit={{.limit}}",
	UpdateJobEndPoint:       "/{{.user}}/job/{{.jobId}}",
	DeleteJobEndPoint:       "/{{.user}}/job/{{.jobId}}",
	ChangeJobSchemaEndPoint: "/{{.user}}/job/{{.jobId}}/schema/{{.schemaId}}/design/{{.designId}}",

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
	//TODO - change it to https
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

	//Handle Error
	ErrorNilCheck(GetFunctionName(HTTPPost), err)
	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	ErrorNilCheck(GetFunctionName(HTTPPost), err)

	return resp.StatusCode, body, err
}

func HTTPPut(url string, msg interface{}, contentType string) (int, []byte, error) {
	putBody, err := json.Marshal(msg)
	if err != nil {
		zap.S().Errorf("error encoding the payload")
		return -1, nil, err
	}

	req, err := http.NewRequest(http.MethodPut, url, bytes.NewBuffer(putBody))
	ErrorNilCheck(GetFunctionName(HTTPPut), err)
	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	client := &http.Client{}
	resp, err := client.Do(req)

	//Handle Error
	ErrorNilCheck(GetFunctionName(HTTPPut), err)
	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	ErrorNilCheck(GetFunctionName(HTTPPut), err)

	return resp.StatusCode, body, err
}

func HTTPGet(url string) (int, []byte, error) {
	resp, err := http.Get(url)

	//handle error
	ErrorNilCheck(GetFunctionName(HTTPGet), err)
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	ErrorNilCheck(GetFunctionName(HTTPGet), err)

	return resp.StatusCode, body, err
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

func CheckStatusCode(code int) error {
	if code >= 400 && code <= 599 {
		return fmt.Errorf("status code: %d", code)
	}

	return nil
}
