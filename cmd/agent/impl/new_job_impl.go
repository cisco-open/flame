package impl

import (
	"os"
	"os/exec"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

//TODO in development
func NewJobInitApp(jobInfo objects.JobNotification) {

	zap.S().Debugf("Init new job. Job details: %v", jobInfo)
	//Step 1 : get information about the job such as schema details

	//Step 2: start the application
	//example python3 main.py --agentIp localhost --name asd --uuid 123asd --agentUuid agent_1
	cmd := exec.Command("python3", "main.py", "--agentIp", "localhost", "--agentUuid", os.Getenv(util.EnvUuid), "--name", "app", "--uuid", "app_1")
	cmd.Stdout = os.Stdout
	err := cmd.Start()
	if err != nil {
		zap.S().Errorf("error starting the application. %v", err)
	}
}
