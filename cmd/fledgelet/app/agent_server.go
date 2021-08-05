package app

import (
	"crypto/md5"
	"encoding/hex"
	"errors"
	"os"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

var Agent AgentInfo

func StartAgent(nsInfo objects.ServerInfo, apiInfo objects.ServerInfo) error {
	// TODO: revisit name and id part;
	// determining name and id can be done through api call
	name, err := os.Hostname()
	if err != nil {
		return err
	}
	hash := md5.Sum([]byte(name))
	uuid := hex.EncodeToString(hash[:])

	if name == "" || uuid == "" {
		zap.S().Error("name or id not determined")
		return errors.New("missing fledgelet name or id")
	}
	Agent = AgentInfo{
		apiInfo: apiInfo,
		nsInfo:  nsInfo,
		name:    name,
		uuid:    uuid,
	}

	zap.S().Infof("Starting fledgelet. name: %s | id: %s", name, uuid)
	InitGRPCService()
	return nil
}
