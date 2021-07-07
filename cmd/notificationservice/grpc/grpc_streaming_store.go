package grpcnotify

import (
	"go.uber.org/zap"
	"google.golang.org/protobuf/types/known/anypb"

	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
)

// SetupAgentStream is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) SetupAgentStream(in *pbNotification.AgentInfo, stream pbNotification.NotificationStreamingStore_SetupAgentStreamServer) error {
	s.addNewClient(in, &stream)
	// the stream should not be killed so we do not return from this server
	// loop infinitely to keep stream alive else this stream will be closed
	for {
	}
	return nil
}

// addNewClient is responsible to add new client to the server map.
func (s *notificationServer) addNewClient(in *pbNotification.AgentInfo, stream *pbNotification.NotificationStreamingStore_SetupAgentStreamServer) {
	uuid := in.GetUuid()
	zap.S().Debugf("Adding new client to the collection | %v", in)
	s.clientStreams[uuid] = stream
	s.clients[uuid] = in
}

// sendNotification is called to send notification to the specific clients.
func (s *notificationServer) sendNotification(clientID string, msg *anypb.Any) {
	zap.S().Debugf("Sending notification to client: %v", clientID)
	stream := s.clientStreams[clientID]

	resp := pbNotification.StreamResponse{
		Type:      pbNotification.StreamResponse_JOB_NOTIFICATION,
		Message:   msg,
		AgentUuid: clientID,
	}

	if err := (*stream).Send(&resp); err != nil {
		zap.S().Errorf("send error %v", err)
	}
}

// broadcastNotification is called to broadcast notification to all the connected clients.
func (s *notificationServer) broadcastNotification(msg *anypb.Any) {
	for clientID := range s.clients {
		s.sendNotification(clientID, msg)
	}
}
