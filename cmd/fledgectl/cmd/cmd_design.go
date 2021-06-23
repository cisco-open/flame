package cmd

import (
	"encoding/json"
	"os"

	objects2 "wwwin-github.cisco.com/fledge/fledge/pkg/objects"
	"wwwin-github.cisco.com/fledge/fledge/pkg/util"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
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
		portNo, err := flags.GetInt64("port")
		//portNo, err := flags.GetString("port")
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
			"user": user,
		}
		url := CreateURI(ip, portNo, CreateDesign, uriMap)

		//Encode the data
		postBody, _ := json.Marshal(objects2.DesignInfo{
			Name:        name,
			Description: description,
		})

		//send post request
		responseBody, _ := HTTPPost(url, postBody, "application/json")
		sb := string(responseBody)
		zap.S().Infof(sb)

		return nil
	},
}

var getDesignCmd = &cobra.Command{
	Use:   "getById",
	Short: "Get design template for the user",
	Long:  "Get design template for the user",
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

		//construct URL
		uriMap := map[string]string{
			"user":     user,
			"designId": dId,
		}
		url := CreateURI(ip, portNo, GetDesign, uriMap)

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

var getDesignsCmd = &cobra.Command{
	Use:   "getList",
	Short: "Get list of all design template for the user",
	Long:  "Get list of all design template for the user",
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

		limit, err := flags.GetString("limit")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":  user,
			"limit": limit,
		}
		url := CreateURI(ip, portNo, GetDesigns, uriMap)

		//send get request
		responseBody, _ := HTTPGet(url)

		//convert the response into list of struct
		var infoList []objects2.DesignInfo
		err = json.Unmarshal(responseBody, &infoList)
		if err != nil {
			return err
		}

		//displaying the output in a table form https://github.com/olekukonko/tablewriter
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"ID", "Name", "Description"})

		for _, v := range infoList {
			table.Append([]string{v.ID.Hex(), v.Name, v.Description})
		}
		table.Render() // Send output

		return nil
	},
}

func init() {
	rootCmd.AddCommand(designCmd)
	designCmd.AddCommand(createDesignCmd, getDesignCmd, getDesignsCmd)

	designCmd.PersistentFlags().Int64P("port", "p", 5000, "listening port for API server")
	designCmd.PersistentFlags().StringP("ip", "i", "0.0.0.0", "IP address for API server")
	designCmd.PersistentFlags().StringP("user", "u", "", "User id")
	//required flags
	designCmd.MarkPersistentFlagRequired("user")

	//local flags for each command
	//CREATE DESIGN
	createDesignCmd.Flags().StringP("name", "n", "sample design", "Design name")
	createDesignCmd.Flags().StringP("desc", "d", "design description placeholder", "Design description")
	//required flags
	createDesignCmd.MarkFlagRequired("name")

	//GET DESIGN
	getDesignCmd.Flags().StringP("designId", "l", "100", "Design id corresponding to the design template information")

	//GET DESIGNS
	getDesignsCmd.Flags().StringP("limit", "l", "100", "List of all the designs by this user")

}
