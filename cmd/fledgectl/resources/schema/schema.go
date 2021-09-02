package schema

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type Params struct {
	Host string
	Port uint16
	User string

	DesignId   string
	SchemaPath string
	Version    string
}

func Create(params Params) error {
	// read schema via conf file
	jsonData, err := ioutil.ReadFile(params.SchemaPath)
	if err != nil {
		return err
	}

	// read schemas from the json file
	schema := openapi.DesignSchema{}
	err = json.Unmarshal(jsonData, &schema)
	if err != nil {
		return err
	}

	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.CreateDesignSchemaEndPoint, uriMap)

	// send post request
	code, _, err := restapi.HTTPPost(url, schema, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a new schema - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Printf("Schema created successfully for design %s\n", params.DesignId)

	return nil
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Version,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.GetDesignSchemaEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get a design schema of version %s for design %s - code: %d, error: %v\n",
			params.Version, params.DesignId, code, err)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		zap.S().Warnf("error while formating json: %v", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.GetDesignSchemasEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get design schemas for design %s - code: %d, error: %v\n", params.DesignId, code, err)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		zap.S().Warnf("error while formating json: %v", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func Update(params Params) error {
	// read schema via conf file
	jsonData, err := ioutil.ReadFile(params.SchemaPath)
	if err != nil {
		return err
	}

	// read schema from the json file
	schema := openapi.DesignSchema{}
	err = json.Unmarshal(jsonData, &schema)
	if err != nil {
		return err
	}

	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Version,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.UpdateDesignSchemaEndPoint, uriMap)

	code, _, err := restapi.HTTPPut(url, schema, "application/json")
	if err != nil && restapi.CheckStatusCode(code) != nil {
		fmt.Printf("error while updating a schema version %s - code: %d, error: %v\n", params.Version, code, err)
		return nil
	}

	fmt.Printf("Updated schema version %s successfully for design %s\n", params.Version, params.DesignId)

	return nil
}
