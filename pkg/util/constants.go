package util

// General
const (
	ALL = "all"
	ID  = "id"
)

// Database
const (
	MONGODB = "mongo"
	MySQL   = "mysql"
)

// Database Fields
const (
	UserId = "userId"
)

// Port numbers
const (
	ApiServerRestApiPort        = 443   // REST API port
	NotificationServiceGrpcPort = 10101 // for notification and push
	ControllerGrpcPort          = 10102 // for handling requests via API server
)
