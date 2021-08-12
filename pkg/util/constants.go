package util

const (
	ProjectName = "fledge"

	//Job
	JobStatus       = "JobStatus"
	AddJobNodes     = "AddJobNodes"
	ChangeJobSchema = "ChangeJobSchema"
	GetBySchemaId   = "GetBySchemaId"

	// General
	ALL          = "all"
	ID           = "id"
	Design       = "design"
	Agents       = "agents"
	Message      = "message"
	Errors       = "errors"
	GenericError = "GenericError"
	InternalUser = "sys"

	// Status
	Status        = "status"
	StatusSuccess = "Success"
	StatusError   = "Error"

	// States
	State           = "state"
	InitState       = "Init"
	StartState      = "Start"
	ReadyState      = "Ready"
	RunningState    = "Running"
	ReloadState     = "Reload"
	StopState       = "Stop"
	TerminatedState = "Terminated"
	ErrorState      = "Error"
	CompletedState  = "Completed"

	// Database
	MONGODB = "mongo"
	MySQL   = "mysql"

	// Database Fields
	//TODO append Field to distinguish the fields
	DBFieldMongoID  = "_id"
	DBFieldUserId   = "userId"
	DesignId        = "designId"
	DBFieldSchemaId = "schemaId"
	DBFieldNodes    = "nodes"

	// Port numbers
	ApiServerRestApiPort        = 10100 // REST API port
	NotificationServiceGrpcPort = 10101 // for notification and push
	ControllerRestApiPort       = 10102 // Controller REST API port
	AgentGrpcPort               = 10103 // for fledgelet - application

	// Service names
	Agent      = "fledgelet"
	ApiServer  = "apiserver"
	Controller = "controller"
	Notifier   = "notifier"
	CliTool    = "fledgectl"
)
