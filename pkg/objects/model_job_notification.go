package objects

// JobNotification - job notification message.
type JobNotification struct {
	Agents []ServerInfo `json:"agents"`

	Job JobInfo `json:"job"`

	SchemaInfo DesignSchema `json:"schemaInfo"`

	NotificationType string `json:"notificationType"`
}
