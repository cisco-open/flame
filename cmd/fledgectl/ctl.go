package main

import (
	"os"

	util2 "wwwin-github.cisco.com/fledge/fledge/pkg/util"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/fledge/fledge/cmd/fledgectl/cmd"
)

func main() {
	loggerMgr := util2.InitZapLog()
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
