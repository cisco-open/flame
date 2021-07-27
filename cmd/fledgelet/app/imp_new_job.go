package app

import (
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func NewJobInitApp(jobInfo objects.JobInfo) {
	zap.S().Debugf("Init new job. Job details: %v", jobInfo)
	//Step 1 : get information about the job such as schema details
	//construct URL
	uriMap := map[string]string{
		"user":  util.InternalUser,
		"jobId": jobInfo.ID,
	}
	url := CreateURI(util.GetJobEndPoint, uriMap)

	//send get request
	responseBody, err := util.HTTPGet(url)
	if err != nil {
		zap.S().Errorf("error while getting job infromation. Cannot intalize the job. %v", err)
		return
	}

	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		zap.S().Errorf("error while unpacking response %v", err)
	} else {
		zap.S().Infof("job details : %v", string(prettyJSON))
	}

	//Step 2: start the application
	//example python3 main.py --agentIp localhost --name asd --uuid 123asd --agentUuid agent_1
	//cmd := exec.Command("python3", "main.py", "--agentIp", "localhost", "--agentUuid", os.Getenv(util.EnvUuid), "--name", "app", "--uuid", "app_1")
	//cmd.Stdout = os.Stdout
	//err := cmd.Start()
	//if err != nil {
	//	zap.S().Errorf("error starting the application. %v", err)
	//}
}
