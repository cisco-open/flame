package app

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"os/exec"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

//TODO remove me after demo-day
var Conf objects.AppConf

func NewJobInit(info objects.AppConf) {
	Conf = info
	job := info.JobInfo
	zap.S().Debugf("Init new job. Job id: %s | design id: %s | schema id: %s | role: %s", job.ID, job.DesignId, job.SchemaId, info.Role)

	//Step 1 : dump the information in application confirmation json file
	err := initApp(info)
	req := objects.AgentStatus{
		UpdateType: util.JobStatus,
		Status:     util.StatusError,
		Message:    "",
	}

	if err != nil {
		zap.S().Errorf("error intializing configuration details for application. %v", err)
		req.Message = err.Error()
	} else {
		//Step 2 : Start Application based on policy
		//default state would be ready because configuration file is created and
		//if it is a non data consumer agent next step would be to start the application
		req.Message = util.ReadyState

		if startAppPolicy(info) {
			state, err := NewJobStart(info)

			if err != nil {
				req.Status = util.StatusError
				req.Message = err.Error()
			} else {
				req.Status = util.StatusSuccess
				req.Message = state
			}
		}
	}

	//Update agents status
	//construct URL
	uriMap := map[string]string{
		"user":    util.InternalUser,
		"jobId":   job.ID,
		"agentId": Agent.uuid,
	}
	url := CreateURI(util.UpdateAgentStatusEndPoint, uriMap)

	//send post request
	zap.S().Debugf("Sending update status call to controller. Current state: %s", req.Message)
	code, response, err := util.HTTPPut(url, req, "application/json")
	if err != nil {
		zap.S().Errorf("error while updating the agent status. %v", err)
	} else {
		zap.S().Debugf("update response code: %d | response: %s", code, string(response))
	}
}

func initApp(info objects.AppConf) error {
	//directory path
	filepath := "/fledge/job/" + info.JobInfo.ID
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		zap.S().Debugf("Creating filepath: %s", filepath)
		os.MkdirAll(filepath, 0700) // Create your file
	}
	confFilePath := filepath + "/conf.json"

	//create a configuration file for the application
	file, _ := json.MarshalIndent(info, "", " ")
	err := ioutil.WriteFile(confFilePath, file, 0644)
	return err
}

func startAppPolicy(info objects.AppConf) bool {
	//will start the application as part of the init if the node is not a data consumer
	for _, role := range info.SchemaInfo.Roles {
		if role.Name == info.Role && !role.IsDataConsumer {
			return true
		}
	}
	return false
}

//NewJobStart starts the application on the agent
func NewJobStart(info objects.AppConf) (string, error) {
	cmd := exec.Command("python3", Conf.Command...)
	zap.S().Debugf("Starting application. Command : %v", cmd)

	//dump output to a log file
	outfile, err := os.Create("./application_" + Agent.name + ".log")
	if err != nil {
		zap.S().Errorf("%v", err)
	}
	cmd.Stdout = outfile
	cmd.Stderr = outfile
	//cmd.Stdout = os.Stdout

	err = cmd.Start()
	if err != nil {
		zap.S().Errorf("error starting the application. %v", err)
		return util.ErrorState, err
	}

	return util.RunningState, nil
}
