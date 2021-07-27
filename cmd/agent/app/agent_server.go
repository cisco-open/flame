package app

import (
	"crypto/md5"
	"encoding/hex"
	"errors"
	"os"

	"go.uber.org/zap"

	grpcagent "wwwin-github.cisco.com/eti/fledge/cmd/agent/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

func StartAgent(nsInfo objects.ServerInfo) error {
	// TODO: revisit name and id part;
	// determining name and id can be done through api call
	name, err := os.Hostname()
	if err != nil {
		return err
	}

	hash := md5.Sum([]byte(name))
	id := hex.EncodeToString(hash[:])

	if name == "" || id == "" {
		zap.S().Error("name or id not determined")
		return errors.New("missing agent name or id")
	}

	zap.S().Infof("Starting agent ... name: %s | id: %s", name, id)
	grpcagent.InitGRPCService(name, id, nsInfo)

	return nil
}
