package objects

type AppConf struct {
	JobInfo JobInfo `json:"jobInfo"`

	SchemaInfo DesignSchema `json:"schemaInfo"`

	Role string `json:"role"`

	//TODO remove me after demo-day
	Command []string `json:"command"`
}
