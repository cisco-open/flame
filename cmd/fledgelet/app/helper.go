package app

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type AgentInfo struct {
	apiInfo objects.ServerInfo
	nsInfo  objects.ServerInfo
	name    string
	uuid    string
}

type AppInfo struct {
	State string
	Conf objects.AppConf
}

func CreateURI(endPoint string, uriMap map[string]string) string {
	return util.CreateURI(Agent.apiInfo.IP, util.ApiServerRestApiPort, endPoint, uriMap)
}
