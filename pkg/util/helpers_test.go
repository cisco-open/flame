package util

import (
	"os/user"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInitZapLog(t *testing.T) {
	logger := InitZapLog("some_service")

	currentUser, err := user.Current()
	assert.Nil(t, err)

	if currentUser.Username == "root" {
		assert.NotNil(t, logger)
	} else {
		assert.Nil(t, logger)
	}
}
