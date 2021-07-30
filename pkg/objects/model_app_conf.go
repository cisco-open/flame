package objects

type AppConf struct {
	JobInfo JobInfo `json:"jobInfo"`

	SchemaInfo DesignSchema `json:"schemaInfo"`

	Role string
}
