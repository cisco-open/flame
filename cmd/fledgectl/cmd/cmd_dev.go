package cmd

import (
	"io/ioutil"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"

	"gopkg.in/yaml.v3"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
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

		portNo, err := flags.GetInt64("port")
		if err != nil {
			return err
		}

		ip, err := flags.GetString("ip")
		if err != nil {
			return err
		}

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		user, err := flags.GetString("userId")
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
			"user": user,
		}
		url := util.CreateURI(ip, portNo, util.JobNodesEndPoint, uriMap)

		//Read schemas from the yaml file
		//var y []objects.ServerInfo
		type Tmp struct {
			Nodes []objects.ServerInfo `json:"nodes,omitempty"`
		}
		y := Tmp{}
		err = yaml.Unmarshal(yamlFile, &y)

		jN := objects.JobNodes{
			DesignId: designId,
			Nodes:    y.Nodes,
		}
		//zap.S().Infof("url : %s | jobId: %s", url, jobId)
		//zap.S().Infof("conf %v", yamlFile)
		//zap.S().Infof("nodes: %v", y.Nodes)

		//send post request
		_, responseBody, err := util.HTTPPost(url, jN, "application/json")
		if err != nil {
			zap.S().Errorf("error while adding a new schema. %v", err)
		} else {
			sb := string(responseBody)
			zap.S().Infof(sb)
		}
		return nil
	},
}

var updateJobNodesCmd = &cobra.Command{
	Use:   "updateNodes",
	Short: "Update existing/Add new information about nodes for job",
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

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		user, err := flags.GetString("userId")
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
			"user": user,
		}
		url := util.CreateURI(ip, portNo, util.JobNodesEndPoint, uriMap)

		//Read schemas from the yaml file
		//var y []objects.ServerInfo
		type Tmp struct {
			Nodes []objects.ServerInfo `json:"nodes,omitempty"`
		}
		y := Tmp{}
		err = yaml.Unmarshal(yamlFile, &y)

		jN := objects.JobNodes{
			DesignId: designId,
			Nodes:    y.Nodes,
		}

		//send post request
		_, responseBody, err := util.HTTPPut(url, jN, "application/json")
		if err != nil {
			zap.S().Errorf("error while adding a new schema. %v", err)
		} else {
			sb := string(responseBody)
			zap.S().Infof(sb)
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(devCmd)
	devCmd.AddCommand(jobNodesCmd, updateJobNodesCmd)

	devCmd.PersistentFlags().Int64P("port", "p", util.ApiServerRestApiPort, "listening port for API server")
	devCmd.PersistentFlags().StringP("ip", "i", "0.0.0.0", "IP address for API server")

	//local flags for each command
	//JOB NODES
	jobNodesCmd.Flags().StringP("designId", "j", "", "Design id")
	jobNodesCmd.Flags().StringP("userId", "u", "", "User id")
	jobNodesCmd.Flags().StringP("conf", "c", "", "Configuration file with schema design")
	jobNodesCmd.MarkFlagRequired("designId")
	jobNodesCmd.MarkFlagRequired("userId")
	jobNodesCmd.MarkFlagRequired("conf")

	//JOB NODES
	updateJobNodesCmd.Flags().StringP("designId", "j", "", "Design id")
	updateJobNodesCmd.Flags().StringP("userId", "u", "", "User id")
	updateJobNodesCmd.Flags().StringP("conf", "c", "", "Configuration file with schema design")
	updateJobNodesCmd.MarkFlagRequired("designId")
	updateJobNodesCmd.MarkFlagRequired("userId")
	updateJobNodesCmd.MarkFlagRequired("conf")
}
