package main

import (
	"os"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/fledgectl/cmd"
	util2 "wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util2.InitZapLog(util2.CliTool)
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
