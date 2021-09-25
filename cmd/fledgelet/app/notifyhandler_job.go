// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package app

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	workDir = "/fledge/work"
)

// startJob starts the application on the agent
func (h *NotifyHandler) startJob(jobId string) {
	zap.S().Infof("Received start job request on job %s", jobId)

	filePaths, err := h.getTask(jobId)
	if err != nil {
		zap.S().Warnf("Failed to download payload: %v", err)
		return
	}

	err = h.prepareTask(filePaths)
	if err != nil {
		zap.S().Warnf("Failed to prepare task")
		return
	}

	// TODO: implement/revise startApp method

	// TODO: implement updateTaskStatus method
}

func (h *NotifyHandler) getTask(jobId string) ([]string, error) {
	// construct URL
	uriMap := map[string]string{
		"jobId":   jobId,
		"agentId": h.agentId,
	}
	url := restapi.CreateURL(h.apiserverEp, restapi.GetTaskEndpoint, uriMap)

	code, taskMap, err := restapi.HTTPGetMultipart(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		errMsg := fmt.Sprintf("Failed to fetch task - code: %d, error: %v", code, err)
		zap.S().Warnf(errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	filePaths := make([]string, 0)
	for fileName, data := range taskMap {
		filePath := filepath.Join("/tmp", fileName)
		err = ioutil.WriteFile(filePath, data, util.FilePerm0755)
		if err != nil {
			zap.S().Warnf("Failed to save %s: %v\n", fileName, err)
			return nil, err
		}

		filePaths = append(filePaths, filePath)

		zap.S().Infof("Downloaded %s successfully\n", fileName)
	}

	return filePaths, nil
}

func (h *NotifyHandler) prepareTask(filePaths []string) error {
	err := os.MkdirAll(workDir, util.FilePerm0755)
	if err != nil {
		return err
	}

	var fileDataList []util.FileData
	var file *os.File
	configFilePath := ""
	configFound := false
	codeFound := false
	for _, filePath := range filePaths {
		if strings.Contains(filePath, util.TaskConfigFile) {
			configFound = true

			configFilePath = filePath
		} else if strings.Contains(filePath, util.TaskCodeFile) {
			codeFound = true

			file, err = os.Open(filePath)
			if err != nil {
				return fmt.Errorf("failed to open %s: %v", filePath, err)
			}

			fileDataList, err = util.UnzipFile(file)
			if err != nil {
				return err
			}
		}
	}

	if !configFound || !codeFound {
		return fmt.Errorf("either %s or %s not found", util.TaskConfigFile, util.TaskCodeFile)
	}

	// copy config file to work directory
	input, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return fmt.Errorf("failed to open config file %s: %v", configFilePath, err)
	}

	dstFilePath := filepath.Join(workDir, util.TaskConfigFile)
	err = ioutil.WriteFile(dstFilePath, input, util.FilePerm0644)
	if err != nil {
		return fmt.Errorf("failed to copy config file: %v", err)
	}

	// copy code files to work directory
	for _, fileData := range fileDataList {
		dirPath := filepath.Join(workDir, filepath.Dir(fileData.FullName))
		err := os.MkdirAll(dirPath, util.FilePerm0755)
		if err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}

		filePath := filepath.Join(dirPath, fileData.BaseName)
		err = ioutil.WriteFile(filePath, []byte(fileData.Data), util.FilePerm0644)
		if err != nil {
			return fmt.Errorf("failed to unzip file %s: %v", filePath, err)
		}
	}

	return nil
}

func (h *NotifyHandler) stopJob(jobId string) {
	zap.S().Infof("not yet implemented; received stop job request on job %s", jobId)
	// h.appInfo.Conf = info
	// job := info.JobInfo
	// zap.S().Debugf("Init new job. Job id: %s | design id: %s | schema id: %s | role: %s", job.Id, job.DesignId, job.SchemaId, info.Role)

	// //Step 1 : dump the information in application confirmation json file
	// err := h.initApp(info)
	// req := openapi.AgentStatus{
	// 	UpdateType: util.JobStatus,
	// 	Status:     "",
	// 	Message:    "",
	// }

	// if err != nil {
	// 	zap.S().Errorf("Failed to initialize configuration details for application: %v", err)
	// 	req.Status = util.StatusError
	// 	req.Message = err.Error()
	// } else {
	// 	//Step 2 : Start Application based on policy
	// 	//default state would be ready because configuration file is created and
	// 	//if it is a non data consumer agent next step would be to start the application
	// 	req.Status = util.StatusSuccess
	// 	req.Message = util.ReadyState

	// 	if h.startAppPolicy(info) {
	// 		state, err := h.startApp(info)

	// 		if err != nil {
	// 			req.Status = util.StatusError
	// 			req.Message = err.Error()
	// 		} else {
	// 			req.Status = util.StatusSuccess
	// 			req.Message = state
	// 		}
	// 	}
	// }

	// h.updateJobStatus(job, req)
}

func (h *NotifyHandler) updateJob(jobId string) (string, error) {
	zap.S().Infof("not yet implemented; received update job request on job %s", jobId)
	return "", nil
}

/*
func (h *NotifyHandler) initApp(info objects.AppConf) error {
	//directory path
	fp := filepath.Join("/fledge/job/", h.agentId, info.JobInfo.Id)
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

func (h *NotifyHandler) startAppPolicy(info objects.AppConf) bool {
	//will start the application as part of the init if the node is not a data consumer
	for _, role := range info.SchemaInfo.Roles {
		if role.Name == info.Role && !role.IsDataConsumer {
			return true
		}
	}
	return false
}

func (h *NotifyHandler) startApp(_ objects.AppConf) (string, error) {
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

func (h *NotifyHandler) updateJobStatus(job openapi.JobInfo, req openapi.AgentStatus) {
	//update application state
	h.appInfo.State = req.Message
	if req.Status == util.StatusError {
		h.appInfo.State = util.ErrorState
	}

	//Update agents status
	//construct URL
	uriMap := map[string]string{
		"user":    util.InternalUser,
		"jobId":   job.Id,
		"agentId": h.agentId,
	}

	url := restapi.CreateURL(h.apiserverEp, restapi.UpdateAgentStatusEndPoint, uriMap)

	//send post request
	zap.S().Debugf("Sending update status call to controller. Current state: %s", req.Message)
	code, response, err := restapi.HTTPPut(url, req, "application/json")
	if err != nil {
		zap.S().Errorf("error while updating the agent status. %v", err)
	} else {
		zap.S().Debugf("update response code: %d | response: %s", code, string(response))
	}
}
*/
