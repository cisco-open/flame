package util

const (
	ProjectName = "fledge"

	// General
	ALL          = "all"
	ID           = "id"
	Design       = "design"
	Agents       = "agents"
	Message      = "message"
	Errors       = "errors"
	Initializing = "Initializing"
	Init         = "Init"
	Start        = "Start"
	Running      = "Running"
	Stop         = "Stop"
	Terminated   = "Terminated"
	Completed    = "Completed"
	GenericError = "GenericError"

	// Database
	MONGODB = "mongo"
	MySQL   = "mysql"

	// Database Fields
	MongoID  = "_id"
	UserId   = "userId"
	DesignId = "designId"

	AddJobNodes         = "AddJobNodes"
	UpdateJobStatus     = "UpdateJobStatus"
	UpdateJobNodeStatus = "UpdateJobNodeStatus"

	// Port numbers
	ApiServerRestApiPort        = 10100 // REST API port
	NotificationServiceGrpcPort = 10101 // for notification and push
	ControllerRestApiPort       = 10102 // Controller REST API port
	AgentGrpcPort               = 10103 // for agent - application

	// Service names
	Agent      = "agent"
	ApiServer  = "apiserver"
	Controller = "controller"
	Notifier   = "notifier"
	CliTool    = "fledgectl"
)
