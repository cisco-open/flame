package util

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"text/template"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/protobuf/types/known/structpb"
)

/*InitZapLog Zap Logger initialization

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

//StructToMapInterface converts any struct interface into map interface using json de-coding/encoding.
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

//URI must always end with a backslash. Make sure the query params are not ended with a backslash
//API END POINTS
const (
	//Design
	CreateDesignEndPoint       = "CREATE_DESIGN"
	GetDesignsEndPoint         = "GET_DESIGNS"
	GetDesignEndPoint          = "GET_DESIGN"
	UpdateDesignSchemaEndPoint = "UPDATE_DESIGN_SCHEMA"
	GetDesignSchemaEndPoint    = "GET_DESIGN_SCHEMA"
	//Job
	SubmitJobEndPoint = "SUBMIT_JOB"
	GetJobEndPoint    = "GET_JOB"
	GetJobsEndPoint   = "GET_JOBS"
	DeleteJobEndPoint = "DELETE_JOB"
	UpdateJobEndPoint = "UPDATE_JOB"
	//Agent
	UpdateAgentStatusEndPoint = "UPDATE_AGENT_STATUS"

	//TODO remove me after prototyping phase is done.
	JobNodesEndPoint = "JOB_NODES"
)

var URI = map[string]string{
	// Design Template
	CreateDesignEndPoint:       "/{{.user}}/design/",
	GetDesignEndPoint:          "/{{.user}}/design/{{.designId}}/",
	GetDesignsEndPoint:         "/{{.user}}/designs/?limit={{.limit}}",
	UpdateDesignSchemaEndPoint: "/{{.user}}/design/{{.designId}}/schema/",
	GetDesignSchemaEndPoint:    "/{{.user}}/design/{{.designId}}/schema/?getType={{.type}}&schemaId={{.schemaId}}",

	//Job
	SubmitJobEndPoint: "/{{.user}}/job/",
	GetJobEndPoint:    "/{{.user}}/job/{{.jobId}}",
	GetJobsEndPoint:   "/{{.user}}/jobs/?getType={{.type}}&designId={{.designId}}&limit={{.limit}}",
	UpdateJobEndPoint: "/{{.user}}/job/{{.jobId}}",
	DeleteJobEndPoint: "/{{.user}}/job/{{.jobId}}",

	//Agent
	UpdateAgentStatusEndPoint: "/{{.user}}/job/{{.jobId}}/agent/{{.agentId}}",

	//TODO remove me after prototyping phase is done.
	JobNodesEndPoint: "/{{.user}}/nodes/",
}

func CreateURI(ip string, portNo int64, endPoint string, inputMap map[string]string) string {
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

func HTTPGet(url string) ([]byte, error) {
	resp, err := http.Get(url)

	//handle error
	ErrorNilCheck(GetFunctionName(HTTPGet), err)
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	ErrorNilCheck(GetFunctionName(HTTPGet), err)

	return body, err
}
