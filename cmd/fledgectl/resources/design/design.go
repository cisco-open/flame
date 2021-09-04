package design

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type Params struct {
	resources.CommonParams

	DesignId string
	Desc     string
	Limit    string
}

func Create(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.CreateDesignEndPoint, uriMap)

	//Encode the data
	postBody := openapi.DesignInfo{
		Id:          params.DesignId,
		Description: params.Desc,
	}

	// send post request
	code, _, err := restapi.HTTPPost(url, postBody, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a new design - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Println("New design created successfully")
	return nil
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.GetDesignEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to retrieve design %s - code: %d, error: %v\n", params.DesignId, code, err)
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
		"user":  params.User,
		"limit": params.Limit,
	}
	url := restapi.CreateURL(params.Host, params.Port, restapi.GetDesignsEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get design templates - code: %d, error: %v\n", code, err)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.DesignInfo{}
	err = json.Unmarshal(responseBody, &infoList)
	if err != nil {
		fmt.Printf("Failed to unmarshal design templates: %v\n", err)
		return nil
	}

	// displaying the output in a table form https://github.com/olekukonko/tablewriter
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Deisgn ID", "Name", "Description"})

	for _, v := range infoList {
		table.Append([]string{v.Id, v.Name, v.Description})
	}

	table.Render() // Send output

	return nil
}
