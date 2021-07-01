package cmd

import (
	"encoding/json"
	"io/ioutil"

	objects2 "wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"

	"gopkg.in/yaml.v3"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var designSchemaCmd = &cobra.Command{
	Use:   "schema",
	Short: "Design Schema Commands",
	Long:  "Design Schema Commands",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var createDesignSchemaCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new design schema",
	Long:  "Create a new design schema",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()
		portNo, err := flags.GetInt64("port")
		if err != nil {
			return err
		}

		ip, err := flags.GetString("ip")
		if err != nil {
			return err
		}

		user, err := flags.GetString("user")
		if err != nil {
			return err
		}

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		conf, err := flags.GetString("conf")
		if err != nil {
			return err
		}

		//read schema via conf file
		yamlFile, err := ioutil.ReadFile(conf)
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":     user,
			"designId": designId,
		}
		url := CreateURI(ip, portNo, UpdateDesignSchema, uriMap)

		//Read schemas from the yaml file
		var y = objects2.DesignSchemas{}
		err = yaml.Unmarshal(yamlFile, &y)

		//Send post request for each schema
		for _, s := range y.Schemas {
			postBody, _ := json.Marshal(s)

			//send post request
			responseBody, _ := HTTPPost(url, postBody, "application/json")
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
		portNo, err := flags.GetInt64("port")
		if err != nil {
			return err
		}

		ip, err := flags.GetString("ip")
		if err != nil {
			return err
		}

		user, err := flags.GetString("user")
		if err != nil {
			return err
		}

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

		//construct URL
		uriMap := map[string]string{
			"user":     user,
			"designId": dId,
			"type":     getType,
			"schemaId": schemaId,
		}
		url := CreateURI(ip, portNo, GetDesignSchema, uriMap)

		//send get request
		responseBody, _ := HTTPGet(url)

		//format the output into prettyJson format
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
	designSchemaCmd.AddCommand(createDesignSchemaCmd, getDesignSchemaCmd)

	designSchemaCmd.PersistentFlags().StringP("designId", "d", "", "Design id")
	designSchemaCmd.MarkPersistentFlagRequired("designId")

	//local flags for each command
	//GET DESIGN SCHEMA
	getDesignSchemaCmd.Flags().StringP("type", "t", "all", "Fetch type for design schema. Options - all/id")
	getDesignSchemaCmd.Flags().StringP("schemaId", "x", "", "If fetch type is ID this flag is used to provide the schema id value")

	//CREATE DESIGN SCHEMA
	createDesignSchemaCmd.Flags().StringP("conf", "c", "", "Configuration file with schema design")
	createDesignSchemaCmd.MarkFlagRequired("conf")
}
