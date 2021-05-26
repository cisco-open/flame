package cmd

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExecuteFailure(t *testing.T) {
	rootCmd.SetArgs([]string{"--unknown_flag"})
	err := Execute()
	assert.NotNil(t, err)
	assert.Equal(t, "unknown flag: --unknown_flag", err.Error())
}
