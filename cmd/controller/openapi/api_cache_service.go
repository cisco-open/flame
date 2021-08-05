package openapi

import "wwwin-github.cisco.com/eti/fledge/pkg/objects"

type CacheCollection struct {
	jobAgents map[string][]objects.ServerInfo
	jobSchema map[string]objects.DesignSchema
}

var Cache CacheCollection

func CacheInit() {
	Cache.jobAgents = make(map[string][]objects.ServerInfo)
	Cache.jobSchema = make(map[string]objects.DesignSchema)
}
