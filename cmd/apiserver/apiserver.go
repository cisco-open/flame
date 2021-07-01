package main

import (
	"os"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/apiserver/cmd"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog() //use zap.S().Infof("") or zap.L().Infof("")
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync() // flushes buffer, if any

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
