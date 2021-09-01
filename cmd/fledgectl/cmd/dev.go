package cmd

import (
	"io/ioutil"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"gopkg.in/yaml.v3"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
)

var devCmd = &cobra.Command{
	Use:   "dev",
	Short: "Dev Commands",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var jobNodesCmd = &cobra.Command{
	Use:   "nodes",
	Short: "Provide information about nodes for job",
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

		//read schema via conf file
		yamlFile, err := ioutil.ReadFile(conf)
		if err != nil {
			return err
		}

		//Read schemas from the yaml file
		//var y []objects.ServerInfo
		type Tmp struct {
			Nodes []openapi.ServerInfo `json:"nodes,omitempty"`
		}
		y := Tmp{}
		err = yaml.Unmarshal(yamlFile, &y)
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user": config.User,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.JobNodesEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		jN := openapi.JobNodes{
			DesignId: designId,
			Nodes:    y.Nodes,
		}

		//send post request
		code, responseBody, err := restapi.HTTPPost(url, jN, "application/json")
		if err != nil {
			zap.S().Errorf("error while adding a new schema. %v", err)
		} else {
			zap.S().Debugf("Response code: %d | response: %s", code, string(responseBody))
			zap.S().Debugf("Successfully added %d # of nodes to designId: %s", len(jN.Nodes), designId)
		}
		return nil
	},
}

var updateJobNodesCmd = &cobra.Command{
	Use:   "updateNodes",
	Short: "Update existing/Add new information about nodes for job",
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

		//read schema via conf file
		yamlFile, err := ioutil.ReadFile(conf)
		if err != nil {
			return err
		}

		//Read schemas from the yaml file
		//var y []objects.ServerInfo
		type Tmp struct {
			Nodes []openapi.ServerInfo `json:"nodes,omitempty"`
		}
		y := Tmp{}
		err = yaml.Unmarshal(yamlFile, &y)
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user": config.User,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.JobNodesEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		jN := openapi.JobNodes{
			DesignId: designId,
			Nodes:    y.Nodes,
		}

		//send post request
		code, responseBody, err := restapi.HTTPPut(url, jN, "application/json")
		if err != nil {
			zap.S().Errorf("error while adding a new schema. %v", err)
		} else {
			zap.S().Debugf("Response code: %d | response: %s", code, string(responseBody))
			zap.S().Debugf("Successfully updated the designId: %s with %d # of nodes", designId, len(jN.Nodes))
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(devCmd)
	devCmd.AddCommand(jobNodesCmd, updateJobNodesCmd)

	//local flags for each command
	//JOB NODES
	jobNodesCmd.Flags().StringP("designId", "", "", "Design id")
	jobNodesCmd.Flags().StringP("conf", "", "", "Configuration file with schema design")
	jobNodesCmd.MarkFlagRequired("designId")
	jobNodesCmd.MarkFlagRequired("conf")

	//JOB NODES
	updateJobNodesCmd.Flags().StringP("designId", "", "", "Design id")
	updateJobNodesCmd.Flags().StringP("conf", "", "", "Configuration file with schema design")
	updateJobNodesCmd.MarkFlagRequired("designId")
	updateJobNodesCmd.MarkFlagRequired("conf")
}
