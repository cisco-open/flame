package grpcctlr

import (
	"context"
	"log"
	"net"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pb "wwwin-github.cisco.com/fledge/fledge/pkg/proto/go"
	"wwwin-github.cisco.com/fledge/fledge/pkg/util"
)

type server struct {
	pb.UnimplementedControllerGrpcStoreServer
}

func (s *server) InitAgent(ctx context.Context, in *pb.ExperimentInfo) (*pb.Conf, error) {
	log.Printf("Received: %v", in)
	return &pb.Conf{ExperimentId: "1"}, nil
}

// StartGRPCService example https://github.com/grpc/grpc-go/blob/master/examples/helloworld/greeter_server/main.go
func StartGRPCService() {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(util.GrpcControllerPort))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	s := grpc.NewServer()
	pb.RegisterControllerGrpcStoreServer(s, &server{})
	zap.S().Infof("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}
