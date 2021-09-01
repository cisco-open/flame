package objects

import "wwwin-github.cisco.com/eti/fledge/pkg/openapi"

// JobNotification - job notification message.
type JobNotification struct {
	Agents []openapi.ServerInfo `json:"agents"`

	Job openapi.JobInfo `json:"job"`

	SchemaInfo openapi.DesignSchema `json:"schemaInfo"`

	NotificationType string `json:"notificationType"`
}
