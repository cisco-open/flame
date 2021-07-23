package grpcagent

import (
	"net"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbAgent "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/agent"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

//agentServer implement the agent grpc service and include - proto unimplemented method and
//maintains list of connected apps & their streams.
type agentServer struct {
	clients       map[string]*pbAgent.AppInfo
	clientStreams map[string]*pbAgent.StreamingStore_SetupAppStreamServer

	pbAgent.UnimplementedStreamingStoreServer
}

func (s *agentServer) init() {
	s.clients = make(map[string]*pbAgent.AppInfo)
	s.clientStreams = make(map[string]*pbAgent.StreamingStore_SetupAppStreamServer)
}

//StartGRPCService starts the agent grpc server and register the corresponding stores implemented by agentServer.
func StartGRPCService() {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(util.AgentGrpcPort))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	// create grpc server
	s := grpc.NewServer()
	server := &agentServer{}
	server.init()

	//register grpc services
	pbAgent.RegisterStreamingStoreServer(s, server)

	zap.S().Infof("Agent GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}

// InitGRPCService starts the controller grpc server and establishes connection with the notification service
func InitGRPCService(nsInfo objects.ServerInfo) {
	go ConnectToNotificationService(nsInfo)
	StartGRPCService()
}
