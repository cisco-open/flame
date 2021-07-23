package app

import (
	"errors"
	"os"

	"go.uber.org/zap"
	grpcagent "wwwin-github.cisco.com/eti/fledge/cmd/agent/grpc"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func StartAgent(nsInfo objects.ServerInfo) error {
	name := os.Getenv(util.EnvName)
	uuid := os.Getenv(util.EnvUuid)
	if name == "" || uuid == "" {
		zap.S().Fatal("name and uuid not declared as environment configuration")
		return errors.New("missing agent name/uuid")
	}
	zap.S().Infof("Starting agent ... name: %s | uuid: %s", name, uuid)
	grpcagent.InitGRPCService(nsInfo)
	return nil
}
