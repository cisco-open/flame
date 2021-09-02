/*
 * Fledge REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/gorilla/mux"
)

// A AgentApiController binds http requests to an api service and writes the service results to the http response
type AgentApiController struct {
	service AgentApiServicer
}

// NewAgentApiController creates a default api controller
func NewAgentApiController(s AgentApiServicer) Router {
	return &AgentApiController{service: s}
}

// Routes returns all of the api route for the AgentApiController
func (c *AgentApiController) Routes() Routes {
	return Routes{
		{
			"UpdateAgentStatus",
			strings.ToUpper("Put"),
			"/{user}/job/{jobId}/agent/{agentId}",
			c.UpdateAgentStatus,
		},
	}
}

// UpdateAgentStatus - Update agent status for job id.
func (c *AgentApiController) UpdateAgentStatus(w http.ResponseWriter, r *http.Request) {
	params := mux.Vars(r)
	user := params["user"]

	jobId := params["jobId"]

	agentId := params["agentId"]

	agentStatus := &AgentStatus{}
	if err := json.NewDecoder(r.Body).Decode(&agentStatus); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	result, err := c.service.UpdateAgentStatus(r.Context(), user, jobId, agentId, *agentStatus)
	// If an error occurred, encode the error with the status code
	if err != nil {
		EncodeJSONResponse(err.Error(), &result.Code, w)
		return
	}
	// If no error, encode the body and the result code
	EncodeJSONResponse(result.Body, &result.Code, w)
}