package config

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/cisco-open/flame/pkg/util"
	"github.com/mcuadros/go-defaults"
	"github.com/spf13/viper"
)

// LoadConfigurations reads configuration from file and environment variables.
func LoadConfigurations[T any](config T, paths ...string) (err error) {
	if !isPointer(config) {
		return errors.New("config value is not pointer")
	}

	// preset the default values before populating the config from the environment variables
	defaults.SetDefaults(config)

	// sets the environment variables found in current shell
	viper.AutomaticEnv()

	for _, path := range paths {
		filePath, fileName, extension := util.SplitPath(path)

		if len(fileName) > 0 {
			if len(extension) == 0 {
				return errors.New("config file missing extension")
			}

			var found bool
			for _, ext := range viper.SupportedExts {
				if ext == extension {
					found = true
					break
				}
			}

			if !found {
				return fmt.Errorf("unsupported config file extension. found %s want %v",
					extension,
					viper.SupportedExts,
				)
			}

			viper.SetConfigFile(path)

			viper.SetConfigType(extension)
		}

		// add the config file path to the viper instance
		viper.AddConfigPath(filePath)

		// read the config file
		err = viper.ReadInConfig()
		if err != nil {
			return err
		}
	}

	err = viper.Unmarshal(config)

	return err
}

// isPointer checks if the given interface (i) is a pointer
func isPointer(i interface{}) bool {
	// Get the kind of the given interface (i)
	kind := reflect.ValueOf(i).Kind()

	// Check if the kind matches to reference
	return kind == reflect.Ptr
}
