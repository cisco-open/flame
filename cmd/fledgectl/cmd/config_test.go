package cmd

import (
	"testing"

	"github.com/spf13/afero"
	"github.com/stretchr/testify/assert"

	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var (
	testData = `
apiserver:
  host: localhost
  port: 10100
user: john
`
)

func TestLoadConfig(t *testing.T) {
	fs = afero.NewMemMapFs()

	configFilePath := "/testConfig"

	config, err := loadConfig(configFilePath)
	assert.NotNil(t, err)
	assert.Nil(t, config)

	// write test data
	err = afero.WriteFile(fs, configFilePath, []byte(testData), util.FilePerm0644)
	assert.Nil(t, err)

	config, err = loadConfig(configFilePath)
	assert.Nil(t, err)
	assert.Equal(t, "localhost", config.ApiServer.Host)
	assert.Equal(t, 10100, config.ApiServer.Port)
	assert.Equal(t, "john", config.User)
}
