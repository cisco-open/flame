package cmd

import (
	"bytes"
	"html/template"
	"io/ioutil"
	"net/http"
	"strconv"

	"wwwin-github.cisco.com/fledge/fledge/pkg/util"

	"go.uber.org/zap"
)

//	METHODS
// - - - - - - - - - - - - - - -
const (
	CreateDesign       = "CREATE_DESIGN"
	GetDesigns         = "GET_DESIGNS"
	GetDesign          = "GET_DESIGN"
	UpdateDesignSchema = "UPDATE_DESIGN_SCHEMA"
	GetDesignSchema    = "GET_DESIGN_SCHEMA"
)

//URI must always end with a backslash. Make sure the query params are not ended with a backslash
var URI = map[string]string{
	// Design Template
	"CREATE_DESIGN":        "/{{.user}}/design/",
	"GET_DESIGN":           "/{{.user}}/design/{{.designId}}/",
	"GET_DESIGNS":          "/{{.user}}/designs/?limit={{.limit}}",
	"UPDATE_DESIGN_SCHEMA": "/{{.user}}/design/{{.designId}}/schema/",
	"GET_DESIGN_SCHEMA":    "/{{.user}}/design/{{.designId}}/schema/?getType={{.type}}&schemaId={{.schemaId}}",
}

func CreateURI(ip string, portNo int64, endPoint string, inputMap map[string]string) string {
	//https://stackoverflow.com/questions/29071212/implementing-dynamic-strings-in-golang
	var template = template.Must(template.New("").Parse(URI[endPoint]))
	buf := bytes.Buffer{}
	template.Execute(&buf, inputMap)
	var uri = buf.String()
	var url = "http://" + ip + ":" + strconv.Itoa(int(portNo)) + uri
	zap.S().Debugf("URL : %s", url)
	return url
}

func HTTPPost(url string, postBody []byte, contentType string) ([]byte, error) {
	responseBody := bytes.NewBuffer(postBody)
	resp, err := http.Post(url, contentType, responseBody)

	//Handle Error
	util.ErrorNilCheck(util.GetFunctionName(HTTPPost), err)
	defer resp.Body.Close()

	//Read the response body
	body, err := ioutil.ReadAll(resp.Body)
	util.ErrorNilCheck(util.GetFunctionName(HTTPPost), err)

	return body, err
}

func HTTPGet(url string) ([]byte, error) {
	resp, err := http.Get(url)

	//handle error
	util.ErrorNilCheck(util.GetFunctionName(HTTPGet), err)
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	util.ErrorNilCheck(util.GetFunctionName(HTTPGet), err)

	return body, err
}
