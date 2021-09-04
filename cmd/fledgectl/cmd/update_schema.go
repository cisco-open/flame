package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/schema"
)

var updateDesignSchemaCmd = &cobra.Command{
	Use:   "schema <version>",
	Short: "Update an existing design schema",
	Long:  "Command to update an existing design schema",
	Args:  cobra.RangeArgs(1, 1),
	RunE: func(cmd *cobra.Command, args []string) error {
		version := args[0]

		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		schemaPath, err := flags.GetString("path")
		if err != nil {
			return err
		}

		params := schema.Params{}
		params.Host = config.ApiServer.Host
		params.Port = config.ApiServer.Port
		params.User = config.User
		params.DesignId = designId
		params.SchemaPath = schemaPath
		params.Version = version

		return schema.Update(params)
	},
}

func init() {
	updateDesignSchemaCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	updateDesignSchemaCmd.MarkPersistentFlagRequired("design")
	updateDesignSchemaCmd.PersistentFlags().StringP("path", "p", "", "Path to schema json file")
	updateDesignSchemaCmd.MarkPersistentFlagRequired("path")
	updateCmd.AddCommand(updateDesignSchemaCmd)
}
