package objects

// JobNotification - job notification message.
type JobNotification struct {
	Agents []ServerInfo `json:"agents"`

	Job JobInfo `json:"job"`

	NotificationType string `json:"notificationType"`
}
