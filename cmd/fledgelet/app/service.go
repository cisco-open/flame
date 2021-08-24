package app

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbAgent "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/agent"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type AgentService struct {
	apiServerInfo objects.ServerInfo
	notifierInfo  objects.ServerInfo
	name          string
	uuid          string
	nHandler      *NotificationHandler
}

func NewAgent(apiserverInfo objects.ServerInfo, notifierInfo objects.ServerInfo) (*AgentService, error) {
	// TODO: revisit name and id part;
	// determining name and id can be done through api call
	name, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	hash := md5.Sum([]byte(name))
	uuid := hex.EncodeToString(hash[:])

	if name == "" || uuid == "" {
		err := fmt.Errorf("missing fledgelet name or id")
		zap.S().Error(err)
		return nil, err
	}

	nHandler := newNotificationHandler(apiserverInfo, notifierInfo, name, uuid)

	agent := &AgentService{
		apiServerInfo: apiserverInfo,
		notifierInfo:  notifierInfo,
		name:          name,
		uuid:          uuid,
		nHandler:      nHandler,
	}

	return agent, nil
}

func (agent *AgentService) Start() error {
	zap.S().Infof("Starting %s... name: %s | id: %s", util.Agent, agent.name, agent.uuid)

	agent.nHandler.start()

	err := agent.startAppServer()

	return err
}

// startAppServer starts the fledgelet grpc server and register the corresponding stores implemented by agentServer.
func (agent *AgentService) startAppServer() error {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(util.AgentGrpcPort))
	if err != nil {
		zap.S().Errorf("Failed to listen grpc server: %v", err)
		return err
	}

	// create grpc server
	s := grpc.NewServer()
	server := &appServer{}
	server.init()

	//register grpc services
	pbAgent.RegisterStreamingStoreServer(s, server)

	zap.S().Infof("Agent GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("Failed to serve: %s", err)
		return err
	}

	return nil
}
