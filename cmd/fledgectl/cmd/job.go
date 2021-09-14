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

package cmd

import (
	"net/http"
	"os"
	"strconv"

	"github.com/olekukonko/tablewriter"
	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/restapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	defaultItemLimit = 100
)

var jobCmd = &cobra.Command{
	Use:   "job",
	Short: "Job Submission Commands",
	Long:  "Job Submission Commands",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

var submitJobCmd = &cobra.Command{
	Use:   "submit",
	Short: "Submit a new job",
	Long:  "Submit a new job",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		designId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		schemaId, err := flags.GetString("schemaId")
		if err != nil {
			return err
		}

		name, err := flags.GetString("name")
		if err != nil {
			return err
		}

		description, err := flags.GetString("desc")
		if err != nil {
			return err
		}

		priority, err := flags.GetString("priority")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user": config.User,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.SubmitJobEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		//Encode the data
		postBody := openapi.JobInfo{
			UserId:      config.User,
			DesignId:    designId,
			SchemaId:    schemaId,
			Name:        name,
			Description: description,
			Priority:    priority,
		}

		//send post request
		code, responseBody, err := restapi.HTTPPost(url, postBody, "application/json")
		if err != nil {
			zap.S().Errorf("error while submiting a new job %v", err)
		}

		if code == http.StatusMultiStatus {
			resp := map[string]interface{}{}
			err := util.ByteToStruct(responseBody, &resp)
			if err != nil {
				zap.S().Errorf("error decoding response %v", err)
			}
			zap.S().Infof("Job request submitted with partial error with job id:%s and failed to notifiy clients: %v",
				resp[util.ID], resp[util.Errors])
		} else {
			resp := map[string]interface{}{}
			_ = util.ByteToStruct(responseBody, &resp)
			zap.S().Infof("Job request successfully submitted.\nJob Id: %s\nDesign Id: %s", resp[util.ID], designId)
			zap.S().Infof("Job Id   : %s", resp[util.ID])
			zap.S().Infof("Design Id: %s", designId)
		}

		return nil
	},
}

var getAllJobsCmd = &cobra.Command{
	Use:   "getAllJobs",
	Short: "Get all jobs for given userId or designId",
	Long:  "Get all jobs for given userId or designId",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		dId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		getType, err := flags.GetString("type")
		if err != nil {
			return err
		}

		limit, err := flags.GetInt32("limit")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":     config.User,
			"designId": dId,
			"type":     getType,
			"limit":    strconv.Itoa(int(limit)),
		}

		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.GetJobsEndPoint, uriMap)

		//send get request
		code, responseBody, err := restapi.HTTPGet(url)

		//handle response
		if err != nil {
			zap.S().Errorf("get jobs request failed %v", err)
			return err
		}

		if err = restapi.CheckStatusCode(code); err != nil {
			return err
		}

		var infoList []openapi.JobInfo
		err = util.ByteToStruct(responseBody, &infoList)
		if err != nil {
			zap.S().Errorf("error while decoding response %v", err)
			return err
		}

		//displaying the output in a table form https://github.com/olekukonko/tablewriter
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"ID", "Name", "Description", "DesignId", "Priority", "Created", "Updated", "Completed"})
		for _, v := range infoList {
			table.Append([]string{v.Id, v.Name, v.Description, v.DesignId, v.Priority})
		}
		table.Render() // Send output
		return nil
	},
}

var changeJobSchemaCmd = &cobra.Command{
	Use:   "changeSchema",
	Short: "Change existing design schema associated with the job",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		jId, err := flags.GetString("jobId")
		if err != nil {
			return err
		}

		sId, err := flags.GetString("schemaId")
		if err != nil {
			return err
		}

		dId, err := flags.GetString("designId")
		if err != nil {
			return err
		}

		//construct URL
		uriMap := map[string]string{
			"user":     config.User,
			"jobId":    jId,
			"schemaId": sId,
			"designId": dId,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.ChangeJobSchemaEndPoint, uriMap)
		printCmdInfo(config.ApiServer.Host, config.ApiServer.Port, url)

		//send get request
		code, responseBody, err := restapi.HTTPPost(url, openapi.JobInfo{}, "application/json")
		if err != nil {
			zap.S().Errorf("error while changing to a new schema. %v", err)
		} else {
			zap.S().Debugf("Response code: %d | response: %s", code, string(responseBody))
			zap.S().Debugf("Successfully changed the schema for the running job.")
			zap.S().Debugf("Job Id:	%s", jId)
			zap.S().Debugf("Design Id: %s", dId)
			zap.S().Debugf("New Schema Id: %s", sId)
		}
		return nil
	},
}

// TODO remove me later - use it to quick test a URL
var testCmd = &cobra.Command{
	Use: "test",
	RunE: func(cmd *cobra.Command, args []string) error {
		flags := cmd.Flags()

		jId, err := flags.GetString("jobId")
		if err != nil {
			return err
		}

		uuid, err := flags.GetString("uuid")
		if err != nil {
			return err
		}

		//Update agents status
		//construct URL
		uriMap := map[string]string{
			"user":    util.InternalUser,
			"jobId":   jId,
			"agentId": uuid,
		}
		url := restapi.CreateURL(config.ApiServer.Host, config.ApiServer.Port, restapi.UpdateAgentStatusEndPoint, uriMap)

		zap.S().Debugf("url .. %s", url)

		req := openapi.AgentStatus{
			UpdateType: util.JobStatus,
			Status:     util.StatusSuccess,
			Message:    util.RunningState,
		}

		//send post request
		code, response, err := restapi.HTTPPut(url, req, "application/json")
		if err != nil {
			zap.S().Errorf("error while updating the agent status. %v", err)
		} else {
			zap.S().Debugf("update response code: %d | response: %s", code, string(response))
		}
		return nil
	},
}

func init() {
	rootCmd.AddCommand(jobCmd)
	jobCmd.AddCommand(submitJobCmd, getAllJobsCmd, changeJobSchemaCmd, testCmd)

	//local flags for each command
	//SUBMIT JOB
	submitJobCmd.Flags().StringP("name", "n", "sample job", "Job name")
	submitJobCmd.Flags().StringP("desc", "e", "job description placeholder", "Job description")
	submitJobCmd.Flags().StringP("priority", "x", "default", "Job priority")
	submitJobCmd.Flags().StringP("designId", "d", "", "Design id")
	submitJobCmd.Flags().StringP("schemaId", "s", "", "Schema id")
	//required flags
	submitJobCmd.MarkFlagRequired("name")
	submitJobCmd.MarkFlagRequired("designId")
	submitJobCmd.MarkFlagRequired("schemaId")

	//GET JOB(s)
	getAllJobsCmd.Flags().StringP("type", "t", "all", "Fetch list of all jobs for given user based on type. Options - all/design")
	getAllJobsCmd.Flags().StringP("designId", "d", "", "Design Id")
	getAllJobsCmd.Flags().Int32P("limit", "l", defaultItemLimit, "Item count to be returned")

	testCmd.Flags().StringP("jobId", "j", "", "Job Id")
	testCmd.Flags().StringP("uuid", "x", "", "Agent UUID")

	//CHANGE JOB SCHEMA
	changeJobSchemaCmd.Flags().StringP("designId", "d", "", "Design id")
	changeJobSchemaCmd.Flags().StringP("jobId", "j", "", "Job id")
	changeJobSchemaCmd.Flags().StringP("schemaId", "s", "", "Schema id")
	//required flags
	changeJobSchemaCmd.MarkFlagRequired("designId")
	changeJobSchemaCmd.MarkFlagRequired("jobId")
	changeJobSchemaCmd.MarkFlagRequired("schemaId")
}
