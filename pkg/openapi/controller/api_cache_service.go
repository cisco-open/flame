package controller

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

type CacheCollection struct {
	jobAgents map[string][]openapi.ServerInfo
	jobSchema map[string]openapi.DesignSchema
}

var Cache CacheCollection

// TODO: implement functions to get values from the cache.
// This way it would be easy to switch current in-memory implementation with another cache library
func CacheInit() {
	Cache.jobAgents = make(map[string][]openapi.ServerInfo)
	Cache.jobSchema = make(map[string]openapi.DesignSchema)
}
