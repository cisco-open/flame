package app

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func (h *NotificationHandler) NewJobInit(info objects.AppConf) {
	h.appInfo.Conf = info
	job := info.JobInfo
	zap.S().Debugf("Init new job. Job id: %s | design id: %s | schema id: %s | role: %s", job.ID, job.DesignId, job.SchemaId, info.Role)

	//Step 1 : dump the information in application confirmation json file
	err := h.initApp(info)
	req := objects.AgentStatus{
		UpdateType: util.JobStatus,
		Status:     "",
		Message:    "",
	}

	if err != nil {
		zap.S().Errorf("Failed to initialize configuration details for application: %v", err)
		req.Status = util.StatusError
		req.Message = err.Error()
	} else {
		//Step 2 : Start Application based on policy
		//default state would be ready because configuration file is created and
		//if it is a non data consumer agent next step would be to start the application
		req.Status = util.StatusSuccess
		req.Message = util.ReadyState

		if h.startAppPolicy(info) {
			state, err := h.startApp(info)

			if err != nil {
				req.Status = util.StatusError
				req.Message = err.Error()
			} else {
				req.Status = util.StatusSuccess
				req.Message = state
			}
		}
	}

	h.updateJobStatus(job, req)
}

//NewJobStart starts the application on the agent
func (h *NotificationHandler) NewJobStart(info objects.AppConf) {
	if h.appInfo.State != util.RunningState {
		req := objects.AgentStatus{
			UpdateType: util.JobStatus,
			Status:     "",
			Message:    "",
		}

		state, err := h.startApp(info)
		if err != nil {
			req.Status = util.StatusError
			req.Message = err.Error()
		} else {
			req.Status = util.StatusSuccess
			req.Message = state
		}
		h.updateJobStatus(info.JobInfo, req)
	}
}

func (h *NotificationHandler) JobReload(info objects.AppConf) (string, error) {
	//TODO - update the appInfo - state/conf etc.
	zap.S().Debugf("Reloading job ...")
	return "", nil
}

func (h *NotificationHandler) initApp(info objects.AppConf) error {
	//directory path
	fp := filepath.Join("/fledge/job/", h.uuid, info.JobInfo.ID)
	if _, err := os.Stat(fp); os.IsNotExist(err) {
		zap.S().Debugf("Creating filepath: %s", fp)
		os.MkdirAll(fp, util.FilePerm0700) // Create your file
	}
	confFilePath := fp + "/conf.json"

	//create a configuration file for the application
	file, _ := json.MarshalIndent(info, "", " ")
	err := ioutil.WriteFile(confFilePath, file, util.FilePerm0644)
	return err
}

func (h *NotificationHandler) startAppPolicy(info objects.AppConf) bool {
	//will start the application as part of the init if the node is not a data consumer
	for _, role := range info.SchemaInfo.Roles {
		if role.Name == info.Role && !role.IsDataConsumer {
			return true
		}
	}
	return false
}

func (h *NotificationHandler) startApp(_ objects.AppConf) (string, error) {
	//TODO only for development purpose. We should have single command command to start all applications example python3 main.py
	//cmd := exec.Command("python3", Conf.Command...)
	//cmd := exec.Command("echo", Conf.Command...)
	cmd := exec.Command(h.appInfo.Conf.Command[0], h.appInfo.Conf.Command[1:]...)
	zap.S().Debugf("Starting application. Command : %v", cmd)

	//dump output to a log file
	outfile, err := os.Create("./application_" + h.name + ".log")
	if err != nil {
		zap.S().Errorf("%v", err)
		return util.ErrorState, err
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

func (h *NotificationHandler) updateJobStatus(job objects.JobInfo, req objects.AgentStatus) {
	//update application state
	h.appInfo.State = req.Message
	if req.Status == util.StatusError {
		h.appInfo.State = util.ErrorState
	}

	//Update agents status
	//construct URL
	uriMap := map[string]string{
		"user":    util.InternalUser,
		"jobId":   job.ID,
		"agentId": h.uuid,
	}

	url := util.CreateURL(h.apiServerInfo.IP, util.ApiServerRestApiPort, util.UpdateAgentStatusEndPoint, uriMap)

	//send post request
	zap.S().Debugf("Sending update status call to controller. Current state: %s", req.Message)
	code, response, err := util.HTTPPut(url, req, "application/json")
	if err != nil {
		zap.S().Errorf("error while updating the agent status. %v", err)
	} else {
		zap.S().Debugf("update response code: %d | response: %s", code, string(response))
	}
}
