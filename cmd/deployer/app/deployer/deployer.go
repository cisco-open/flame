// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package deployer

import (
	"fmt"
	"sync"
)

const (
	AKE    = "ake"    // Microsoft’s Azure Kubernetes Service
	EKS    = "eks"    // Amazon’s Elastic Kubernetes Service
	GKE    = "gke"    // Google’s Kubernetes Engine
	K8S    = "k8s"    // vanilla kubernetes
	DOCKER = "docker" // docker or docker compose; only for local dev
)

var (
	envLock sync.Mutex
)

type Deployer interface {
	Initialize(string, string) error
	Install(string, string) error
	Uninstall(string) error
	List() error
	MonitorTask(jobId, taskId string)
	DeleteTaskFromMonitoring(taskId string)
	GetMonitoredPodStatuses() (map[string]TaskHealthDetails, error)
}

func NewDeployer(platform string) (Deployer, error) {
	// TODO: support other platforms: AKE, EKS, GKE, etc.
	switch platform {
	case K8S:
		return NewK8sDeployer()
	case DOCKER:
		return NewDockerDeployer()
	case AKE:
		fallthrough
	case EKS:
		fallthrough
	case GKE:
		fallthrough
	default:
		return nil, fmt.Errorf("unknown platform: %s", platform)
	}
}
