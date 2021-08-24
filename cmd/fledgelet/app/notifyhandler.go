package app

import (
	"context"
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type NotificationHandler struct {
	apiServerInfo objects.ServerInfo
	notifierInfo  objects.ServerInfo
	name          string
	uuid          string

	stream  pbNotification.NotificationStreamingStore_SetupAgentStreamClient
	appInfo AppInfo
}

type AppInfo struct {
	State string
	Conf  objects.AppConf
}

func newNotificationHandler(apiSvrInfo objects.ServerInfo, notifierInfo objects.ServerInfo, name string, uuid string) *NotificationHandler {
	return &NotificationHandler{
		apiServerInfo: apiSvrInfo,
		notifierInfo:  notifierInfo,
		name:          name,
		uuid:          uuid,
	}
}

// start connects to the notifier via grpc and handles notifications from the notifier
func (h *NotificationHandler) start() {
	go h.do_start()
}

func (h *NotificationHandler) do_start() {
	for {
		expBackoff := backoff.NewExponentialBackOff()
		expBackoff.MaxElapsedTime = 5 * time.Minute // max wait time: 5 minutes
		err := backoff.Retry(h.connect, expBackoff)
		if err != nil {
			zap.S().Fatalf("Cannot connect with notifier: %v", err)
		}

		h.do()
	}
}

func (h *NotificationHandler) connect() error {
	// dial server
	conn, err := grpc.Dial(h.notifierInfo.GetAddress(), grpc.WithInsecure())
	if err != nil {
		zap.S().Debugf("Cannot connect with notifier: %v", err)
		return err
	}

	client := pbNotification.NewNotificationStreamingStoreClient(conn)
	in := &pbNotification.AgentInfo{
		Uuid: h.uuid,
		Name: h.name,
	}

	// setup notification stream
	stream, err := client.SetupAgentStream(context.Background(), in)
	if err != nil {
		zap.S().Debugf("Open stream error: %v", err)
		return err
	}

	h.stream = stream
	zap.S().Infof("Connected with notifier at %v", h.notifierInfo)

	return nil
}

func (h *NotificationHandler) do() {
	for {
		resp, err := h.stream.Recv()
		if err != nil {
			zap.S().Errorf("Failed to receive notification: %v", err)
			break
		}

		h.dealWith(resp)
	}

	zap.S().Info("Disconnected from notifier")
}

//newNotification acts as a handler and calls respective functions based on the response type to act on the received notifications.
func (h *NotificationHandler) dealWith(in *pbNotification.StreamResponse) {
	jobMsg := objects.AppConf{}
	err := util.ProtoStructToStruct(in.GetMessage(), &jobMsg)
	if err != nil {
		zap.S().Errorf("error processing the job request. %v", err)
	} else {
		switch in.GetType() {
		case pbNotification.StreamResponse_JOB_NOTIFICATION_INIT:
			h.NewJobInit(jobMsg)
		case pbNotification.StreamResponse_JOB_NOTIFICATION_START:
			h.NewJobStart(jobMsg)
		case pbNotification.StreamResponse_JOB_NOTIFICATION_RELOAD:
			h.JobReload(jobMsg)
		default:
			zap.S().Errorf("Invalid message type: %s", in.GetType())
		}
	}
}
