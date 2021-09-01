package objects

import "wwwin-github.cisco.com/eti/fledge/pkg/openapi"

type AppConf struct {
	JobInfo openapi.JobInfo `json:"jobInfo"`

	SchemaInfo openapi.DesignSchema `json:"schemaInfo"`

	Role string `json:"role"`

	//TODO remove me after demo-day
	Command []string `json:"command"`
}
