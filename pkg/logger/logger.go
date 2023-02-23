package logger

import (
	"github.com/cisco-open/flame/pkg/config"
	"github.com/cisco-open/flame/pkg/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

// Config defines the configuration values for creating a new logger
type Config struct {
	// Environment indicates the environment of the application
	Environment models.Environment `mapstructure:"environment" default:"production"`
}

// New creates a new logger based on the passed in configuration
func New(i *do.Injector) (*zap.Logger, error) {
	cfg := &Config{}

	err := config.LoadConfigurations(cfg) // Load the configuration values from external sources
	if err != nil {
		return nil, err
	}

	var logger *zap.Logger
	if cfg.Environment == models.Production { // Use production level logging if the Environment is production
		logger, err = zap.NewProduction()
		if err != nil {
			return nil, err
		}
	} else { // Otherwise use development level logging
		logger, err = zap.NewDevelopment()
		if err != nil {
			return nil, err
		}
	}

	return logger, nil
}
