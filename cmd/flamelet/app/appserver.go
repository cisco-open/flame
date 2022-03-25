// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package app

import (
	"time"

	"go.uber.org/zap"

	pbAgent "github.com/cisco-open/flame/pkg/proto/go/agent"
)

// appServer implement the flamelet grpc service and include - proto unimplemented method and
// maintains list of connected apps & their streams.
type appServer struct {
	clients       map[string]*pbAgent.AppInfo
	clientStreams map[string]*pbAgent.StreamingStore_SetupAppStreamServer

	pbAgent.UnimplementedStreamingStoreServer
}

func (s *appServer) init() {
	s.clients = make(map[string]*pbAgent.AppInfo)
	s.clientStreams = make(map[string]*pbAgent.StreamingStore_SetupAppStreamServer)
}

// SetupAppStream is called by the application to subscribe to the flamelet.
// Adds the client to the server client map and stores the client stream.
func (s *appServer) SetupAppStream(in *pbAgent.AppInfo, stream pbAgent.StreamingStore_SetupAppStreamServer) error {
	s.addNewClient(in, &stream)
	// the stream should not be killed so we do not return from this server
	// loop infinitely to keep stream alive else this stream will be closed
	//TODO avoid for loop and use channel. Working on it and will submit it as next PR
	for {
		time.Sleep(time.Second)
	}
}

// addNewClient is responsible to add new client to the server map.
func (s *appServer) addNewClient(in *pbAgent.AppInfo, stream *pbAgent.StreamingStore_SetupAppStreamServer) {
	uuid := in.GetUuid()
	zap.S().Debugf("Adding new client to the collection | %v", in)
	s.clientStreams[uuid] = stream
	s.clients[uuid] = in
}

/*
// pushNotification is called to push notification to the specific applications.
func (s *appServer) pushNotification(clientID string, notifyType pbAgent.StreamResponse_ResponseType, in interface{}) error {
	zap.S().Debugf("Sending notification to client: %v", clientID)

	//Step 1 - create notification object
	m, err := util.StructToMapInterface(in)
	if err != nil {
		zap.S().Errorf("error converting notification object into map interface. %v", err)
		return err
	}
	details, err := structpb.NewStruct(m)
	if err != nil {
		zap.S().Errorf("error creating proto struct. %v", err)
		return err
	}

	//Step 2 - send notification
	return s.send(clientID, notifyType, details)
}
*/

/*
//sendNotifications send notification to list of clients in on go
func (s *appServer) pushNotifications(clients []string, notifyType pbAgent.StreamResponse_ResponseType,
	in interface{}) map[string]interface{} {
	//create single notification object since same object is sent to all the clients
	m, err := util.StructToMapInterface(in)
	eList := map[string]interface{}{}
	if err != nil {
		zap.S().Errorf("error converting notification object into map interface. %v", err)
		eList[util.GenericError] = err
		return eList
	}
	details, err := structpb.NewStruct(m)
	if err != nil {
		zap.S().Errorf("error creating proto struct. %v", err)
		eList[util.GenericError] = err
		return eList
	}

	//iterate and send notification to all the clients
	for _, clientID := range clients {
		err = s.send(clientID, notifyType, details)
		if err != nil {
			eList[clientID] = err
		}
	}

	//if no error
	if len(eList) == 0 {
		eList = nil
	}
	return eList
}

// broadcastNotification is called to broadcast notification to all the connected clients.
func (s *appServer) broadcastNotification(notifyType pbAgent.StreamResponse_ResponseType, in interface{}) map[string]interface{} {
	var clients []string
	for clientID := range s.clients {
		clients = append(clients, clientID)
	}
	return s.pushNotifications(clients, notifyType, in)
}

//send implements helper method to push notification to the corresponding streams
func (s *appServer) send(clientID string, notifyType pbAgent.StreamResponse_ResponseType, msg *structpb.Struct) error {
	stream := s.clientStreams[clientID]

	if stream == nil {
		zap.S().Errorf("clientId %s is not registered with the notifiaction service", clientID)
		return errors.New("client not registered with the service")
	}

	resp := pbAgent.StreamResponse{
		Type:    notifyType,
		Message: msg,
	}

	err := (*stream).Send(&resp)
	if err != nil {
		zap.S().Errorf("notification send error. clientId: %s | %v", clientID, err)
	}
	return err
}
*/
