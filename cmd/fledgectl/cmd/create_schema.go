package cmd

import (
	"github.com/spf13/cobra"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/resources/schema"
)

var createDesignSchemaCmd = &cobra.Command{
	Use:   "schema",
	Short: "Create a new design schema",
	Long:  "Command to create a new design schema",
	Args:  cobra.RangeArgs(0, 0),
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("design")
		if err != nil {
			return err
		}

		schemaPath, err := flags.GetString("path")
		if err != nil {
			return err
		}

		params := schema.Params{
			Host: config.ApiServer.Host,
			Port: config.ApiServer.Port,
			User: config.User,

			DesignId:   designId,
			SchemaPath: schemaPath,
		}

		return schema.Create(params)
	},
}

func init() {
	createDesignSchemaCmd.PersistentFlags().StringP("design", "d", "", "Design ID")
	createDesignSchemaCmd.MarkPersistentFlagRequired("design")
	createDesignSchemaCmd.PersistentFlags().StringP("path", "p", "", "Path to schema json file")
	createDesignSchemaCmd.MarkPersistentFlagRequired("path")
	createCmd.AddCommand(createDesignSchemaCmd)
}
