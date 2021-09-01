package cmd

import (
	"encoding/json"
	"io/ioutil"

	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var designSchemaCmd = &cobra.Command{
	Use:   "schema",
	Short: "Design Schema Commands",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var createDesignSchemaCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new design schema",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		conf, err := flags.GetString("conf")
		if err != nil {
			return err
		}

		// read schema via conf file
		jsonData, err := ioutil.ReadFile(conf)
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
			"user":     config.User,
			"designId": designId,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.UpdateDesignSchemaEndPoint, uriMap)

		// send post request
		_, responseBody, err := restapi.HTTPPost(url, schema, "application/json")
		if err != nil {
			zap.S().Errorf("Failed to add a new schema: %v", err)
		} else {
			sb := string(responseBody)
			zap.S().Infof(sb)
		}

		return nil
	},
}

var updateDesignSchemaCmd = &cobra.Command{
	Use:   "update",
	Short: "Update existing design schema",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		schemaId, err := flags.GetString("schemaId")
		if err != nil {
			return err
		}

		conf, err := flags.GetString("conf")
		if err != nil {
			return err
		}

		// read schema via conf file
		jsonData, err := ioutil.ReadFile(conf)
		if err != nil {
			return err
		}

		// read schema from the json file
		schema := openapi.DesignSchema{}
		err = json.Unmarshal(jsonData, &schema)
		if err != nil {
			return err
		}

		schema.Id = schemaId

		//construct URL
		uriMap := map[string]string{
			"user":     config.User,
			"designId": designId,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.UpdateDesignSchemaEndPoint, uriMap)

		_, responseBody, err := restapi.HTTPPost(url, schema, "application/json")
		if err != nil {
			zap.S().Errorf("error while adding a new schema. %v", err)
		} else {
			sb := string(responseBody)
			zap.S().Infof(sb)
		}

		return nil
	},
}

var getDesignSchemaCmd = &cobra.Command{
	Use:   "get",
	Short: "Get design list or specific design schema for the user",
	Long:  "Get design list or specific design schema for the user",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		dId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		getType, err := flags.GetString("type")
		if err != nil {
			return err
		}

		schemaId, err := flags.GetString("schemaId")
		if err != nil {
			return err
		}

		// construct URL
		uriMap := map[string]string{
			"user":     config.User,
			"designId": dId,
			"type":     getType,
			"schemaId": schemaId,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.GetDesignSchemaEndPoint, uriMap)

		// send get request
		responseBody, _ := restapi.HTTPGet(url)

		// format the output into prettyJson format
		prettyJSON, err := util.FormatJSON(responseBody)
		if err != nil {
			zap.S().Errorf("error while formating json %v", err)
			zap.S().Infof(string(responseBody))
		} else {
			zap.S().Infof(string(prettyJSON))
		}

		return nil
	},
}

func init() {
	designCmd.AddCommand(designSchemaCmd)
	designSchemaCmd.AddCommand(createDesignSchemaCmd, getDesignSchemaCmd, updateDesignSchemaCmd)

	designSchemaCmd.PersistentFlags().StringP("designId", "d", "", "Design id")
	designSchemaCmd.MarkPersistentFlagRequired("designId")

	//local flags for each command
	//GET DESIGN SCHEMA
	getDesignSchemaCmd.Flags().StringP("type", "t", "all", "Fetch type for design schema. Options - all/id")
	getDesignSchemaCmd.Flags().StringP("schemaId", "x", "", "If fetch type is ID this flag is used to provide the schema id value")

	//CREATE DESIGN SCHEMA
	createDesignSchemaCmd.Flags().StringP("conf", "c", "", "Configuration file with schema design")
	createDesignSchemaCmd.MarkFlagRequired("conf")

	updateDesignSchemaCmd.Flags().StringP("schemaId", "s", "", "Schema Id that is required to be updated")
	updateDesignSchemaCmd.Flags().StringP("conf", "c", "", "Configuration file with schema design")
	updateDesignSchemaCmd.MarkFlagRequired("schemaId")
	updateDesignSchemaCmd.MarkFlagRequired("conf")
}
