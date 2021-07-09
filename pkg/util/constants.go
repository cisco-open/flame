package util

const (
	ProjectName = "fledge"

	// General
	ALL = "all"
	ID  = "id"

	// Database
	MONGODB = "mongo"
	MySQL   = "mysql"

	// Database Fields
	UserId = "userId"

	// Port numbers
	ApiServerRestApiPort        = 443   // REST API port
	NotificationServiceGrpcPort = 10101 // for notification and push
	ControllerGrpcPort          = 10102 // for handling requests via API server

	// Service names
	Agent      = "agent"
	ApiServer  = "apiserver"
	CliTool    = "fledgectl"
	Controller = "controller"
	Notifier   = "notifier"
)
