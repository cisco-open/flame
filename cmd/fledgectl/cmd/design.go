package cmd

import (
	"encoding/json"
	"os"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var designCmd = &cobra.Command{
	Use:   "design",
	Short: "Design Commands",
	Long:  "Design Commands",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var createDesignCmd = &cobra.Command{
	Use:   "create",
	Short: "Create a new design template",
	Long:  "Create a new design template",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		name, err := flags.GetString("name")
		if err != nil {
			return err
		}

		description, err := flags.GetString("desc")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user": config.User,
		}
		url := util.CreateURL(config.ApiServer.Host, config.ApiServer.Port, util.CreateDesignEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		//Encode the data
		postBody := openapi.DesignInfo{
			Name:        name,
			Description: description,
		}

		//send post request
		code, responseBody, err := util.HTTPPost(url, postBody, "application/json")
		if err != nil {
			zap.S().Errorf("error while creating a new design templated. %v", err)
			return err
		}
		zap.S().Infof("Code: %d | Response: %s", code, string(responseBody))
		zap.S().Infof("New design created.")

		return nil
	},
}

var getDesignCmd = &cobra.Command{
	Use:   "getById",
	Short: "Get design template for the user",
	Long:  "Get design template for the user",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		dId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":     config.User,
			"designId": dId,
		}
		url := util.CreateURL(config.ApiServer.Host, config.ApiServer.Port, util.GetDesignEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

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

var getDesignsCmd = &cobra.Command{
	Use:   "getList",
	Short: "Get list of all design template for the user",
	Long:  "Get list of all design template for the user",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		limit, err := flags.GetString("limit")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":  config.User,
			"limit": limit,
		}
		url := util.CreateURL(config.ApiServer.Host, config.ApiServer.Port, util.GetDesignsEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		//send get request
		responseBody, _ := util.HTTPGet(url)

		//convert the response into list of struct
		var infoList []openapi.DesignInfo
		err = json.Unmarshal(responseBody, &infoList)
		if err != nil {
			return err
		}

		//displaying the output in a table form https://github.com/olekukonko/tablewriter
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"ID", "Name", "Description"})

		for _, v := range infoList {
			table.Append([]string{v.Id, v.Name, v.Description})
		}
		table.Render() // Send output

		return nil
	},
}

func init() {
	rootCmd.AddCommand(designCmd)
	designCmd.AddCommand(createDesignCmd, getDesignCmd, getDesignsCmd)

	//local flags for each command
	//CREATE DESIGN
	createDesignCmd.Flags().StringP("name", "n", "sample design", "Design name")
	createDesignCmd.Flags().StringP("desc", "d", "design description placeholder", "Design description")
	//required flags
	createDesignCmd.MarkFlagRequired("name")

	//GET DESIGN
	getDesignCmd.Flags().StringP("designId", "l", "", "Design id corresponding to the design template information")
	getDesignCmd.MarkFlagRequired("designId")

	//GET DESIGNS
	getDesignsCmd.Flags().StringP("limit", "l", "100", "List of all the designs by this user")
}
