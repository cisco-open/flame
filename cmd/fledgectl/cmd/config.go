package cmd

import (
	"github.com/spf13/afero"
	"gopkg.in/yaml.v3"
)

type Config struct {
	ApiServer ApiServer `yaml:"apiserver"`
	User      string    `yaml:"user"`
}

type ApiServer struct {
	Host string `yaml:"host"`
	Port uint16 `yaml:"port"`
}

var fs afero.Fs

func init() {
	fs = afero.NewOsFs()
}

func loadConfig(configPath string) (*Config, error) {
	data, err := afero.ReadFile(fs, configPath)
	if err != nil {
		return nil, err
	}

	cfg := &Config{}

	err = yaml.Unmarshal(data, cfg)
	if err != nil {
		return nil, err
	}

	return cfg, nil
}
