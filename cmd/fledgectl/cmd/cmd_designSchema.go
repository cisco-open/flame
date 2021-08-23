package cmd

import (
	"io/ioutil"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"

	"gopkg.in/yaml.v3"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
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

		// read schemas from the yaml file
		y := objects.DesignSchemas{}
		err = yaml.Unmarshal(yamlFile, &y)
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":     user,
			"designId": designId,
		}
		url := util.CreateURI(ip, portNo, util.UpdateDesignSchemaEndPoint, uriMap)

		//Send post request for each schema
		for _, s := range y.Schemas {
			//postBody, _ := json.Marshal(s)

			//send post request
			_, responseBody, err := util.HTTPPost(url, s, "application/json")
			if err != nil {
				zap.S().Errorf("error while adding a new schema. %v", err)
			} else {
				sb := string(responseBody)
				zap.S().Infof(sb)
			}
		}
		return nil
	},
}

var updateDesignSchemaCmd = &cobra.Command{
	Use:   "update",
	Short: "Update existing design schema",
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

		schemaId, err := flags.GetString("schemaId")
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

		// read schemas from the yaml file
		schema := objects.DesignSchema{}
		err = yaml.Unmarshal(yamlFile, &schema)
		if err != nil {
			return err
		}

		schema.ID = schemaId

		//construct URL
		uriMap := map[string]string{
			"user":     user,
			"designId": designId,
		}
		url := util.CreateURI(ip, portNo, util.UpdateDesignSchemaEndPoint, uriMap)

		_, responseBody, err := util.HTTPPost(url, schema, "application/json")
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
		url := util.CreateURI(ip, portNo, util.GetDesignSchemaEndPoint, uriMap)

		//send get request
		responseBody, _ := util.HTTPGet(url)

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
