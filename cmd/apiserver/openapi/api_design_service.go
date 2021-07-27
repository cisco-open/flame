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
	"context"
	"errors"
	"net/http"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// DesignApiService is a service that implents the logic for the DesignApiServicer
// This service should implement the business logic for every endpoint for the DesignApi API.
// Include any external packages or services that will be required by this service.
type DesignApiService struct {
}

// NewDesignApiService creates a default api service
func NewDesignApiService() DesignApiServicer {
	return &DesignApiService{}
}

// CreateDesign - Create a new design template.
func (s *DesignApiService) CreateDesign(ctx context.Context, user string, designInfo objects.DesignInfo) (ImplResponse, error) {
	//TODO input validation
	zap.S().Debugf("New design request recieved for user: %s | designInfo: %v", user, designInfo)

	//create controller request
	uriMap := map[string]string{
		"user": user,
	}
	url := CreateURI(util.CreateDesignEndPoint, uriMap)

	//send post request
	_, _, err := util.HTTPPost(url, designInfo, "application/json")

	//response to the user
	if err != nil {
		return Response(http.StatusInternalServerError, nil), errors.New("create new design request failed")
	}
	return Response(http.StatusCreated, nil), nil
}

// GetDesign - Get design template information
func (s *DesignApiService) GetDesign(ctx context.Context, user string, designId string) (ImplResponse, error) {
	//TODO input validation
	zap.S().Debugf("Get design template information for user: %s | designId: %s", user, designId)

	//create controller request
	uriMap := map[string]string{
		"user":     user,
		"designId": designId,
	}
	url := CreateURI(util.GetDesignEndPoint, uriMap)

	//send get request
	responseBody, err := util.HTTPGet(url)

	//response to the user
	if err != nil {
		return Response(http.StatusInternalServerError, nil), errors.New("get design template information request failed")
	}
	resp := objects.Design{}
	err = util.ByteToStruct(responseBody, &resp)
	return Response(http.StatusOK, resp), err
}