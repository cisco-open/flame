package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/schema"
)

var getDesignSchemaCmd = &cobra.Command{
	Use:   "schema [version]",
	Short: "Get a design schema",
	Long:  "This comand retrieves a design schema",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		version := args[0]

		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		params := schema.Params{
			Host: config.ApiServer.Host,
			Port: config.ApiServer.Port,
			User: config.User,

			DesignId: designId,
			Version:  version,
		}

		return schema.Get(params)
	},
}

var getDesignSchemasCmd = &cobra.Command{
	Use:   "schemas",
	Short: "Get design schemas",
	Long:  "This comand retrieves schemas of a design",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		params := schema.Params{
			Host: config.ApiServer.Host,
			Port: config.ApiServer.Port,
			User: config.User,

			DesignId: designId,
		}

		return schema.GetMany(params)
	},
}

func init() {
	getDesignSchemaCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	getDesignSchemaCmd.MarkPersistentFlagRequired("design")
	getDesignSchemasCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	getDesignSchemasCmd.MarkPersistentFlagRequired("design")
	getCmd.AddCommand(getDesignSchemaCmd, getDesignSchemasCmd)
}
