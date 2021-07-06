package main

import (
	"context"
	"strconv"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pb "wwwin-github.cisco.com/eti/fledge/pkg/proto/go"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog()
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	conn, err := grpc.Dial("localhost:"+strconv.Itoa(util.ControllerGrpcPort), grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		zap.S().Fatalf("did not connect: %v", err)
	}

	c := pb.NewControllerGrpcStoreClient(conn)

	// Contact the server and print out its response.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r, err := c.InitAgent(ctx, &pb.ExperimentInfo{ExperimentId: "23023432asd"})
	if err != nil {
		zap.S().Fatalf("could not greet: %v", err)
	}
	zap.S().Infof("Greeting: %s", r.GetExperimentId())
}
