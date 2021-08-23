package openapi

import "wwwin-github.cisco.com/eti/fledge/pkg/objects"

type CacheCollection struct {
	jobAgents map[string][]objects.ServerInfo
	jobSchema map[string]objects.DesignSchema
}

var Cache CacheCollection

// TODO: implement functions to get values from the cache.
// This way it would be easy to switch current in-memory implementation with another cache library
func CacheInit() {
	Cache.jobAgents = make(map[string][]objects.ServerInfo)
	Cache.jobSchema = make(map[string]objects.DesignSchema)
}
