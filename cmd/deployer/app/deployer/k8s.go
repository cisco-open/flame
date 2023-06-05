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
	"context"
	"fmt"
	"os"
	"sort"
	"time"

	"go.uber.org/zap"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/cli"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const podTimeout = time.Second * 5

type TaskHealthDetails struct {
	JobID        string       `json:"job"`
	TaskID       string       `json:"task"`
	Status       v1.PodStatus `json:"status"`
	JobName      string       `json:"job_name"`
	CreationTime time.Time    `json:"creation_time"`
	UnknownCount int
}

type K8sDeployer struct {
	clusterName string // Currently not in use
	namespace   string

	actionConfig action.Configuration

	// Pods to monitor that are running until it's finished
	// taskID => task details
	DeployedTasks map[string]TaskHealthDetails
}

func NewK8sDeployer() (*K8sDeployer, error) {
	return &K8sDeployer{
		DeployedTasks: make(map[string]TaskHealthDetails),
	}, nil
}

func (deployer *K8sDeployer) getEnvSettings(namespace string) (*cli.EnvSettings, error) {
	envLock.Lock()
	defer envLock.Unlock()

	err := os.Setenv("HELM_NAMEPSACE", namespace)
	if err != nil {
		return nil, err
	}

	envSettings := cli.New()
	err = os.Unsetenv("HELM_NAMEPSACE")
	if err != nil {
		return nil, err
	}

	return envSettings, nil
}

func (deployer *K8sDeployer) Initialize(clusterName string, namespace string) error {
	deployer.clusterName = clusterName
	deployer.namespace = namespace

	settings, err := deployer.getEnvSettings(namespace)
	if err != nil {
		return err
	}

	actionConfig := new(action.Configuration)

	err = actionConfig.Init(settings.RESTClientGetter(), settings.Namespace(), os.Getenv("HELM_DRIVER"), zap.S().Debugf)
	if err != nil {
		return err
	}

	deployer.actionConfig = *actionConfig

	return nil
}

func (deployer *K8sDeployer) Install(releaseName string, chartPath string) error {
	chart, err := loader.Load(chartPath)
	if err != nil {
		return err
	}

	installObj := action.NewInstall(&deployer.actionConfig)
	installObj.Namespace = deployer.namespace
	installObj.ReleaseName = releaseName
	release, err := installObj.Run(chart, nil)
	if err != nil {
		if release != nil && release.Info != nil {
			zap.S().Errorf("Release %s failed with status : %v", releaseName, release.Info.Status)
		}
		return err
	}

	zap.S().Infof("Successfully installed release: %s", release.Name)

	return nil
}

func (deployer *K8sDeployer) Uninstall(releaseName string) error {
	uninstallObj := action.NewUninstall(&deployer.actionConfig)
	resp, err := uninstallObj.Run(releaseName)
	if err != nil {
		return err
	}

	zap.S().Infof("Successfully removed %s", resp.Release.Name)
	return nil
}

func (deployer *K8sDeployer) List() error {
	listObj := action.NewList(&deployer.actionConfig)
	listObj.StateMask ^= action.ListFailed // clear failed releases
	listObj.Run()

	return nil
}

func (deployer *K8sDeployer) MonitorTask(jobId, taskId string) {
	jobName := fmt.Sprintf("%s-agent-%s", deployer.namespace, taskId)

	zap.S().Debugf("start monitoring pod of job %s", jobName)

	deployer.DeployedTasks[taskId] = TaskHealthDetails{
		JobID:   jobId,
		TaskID:  taskId,
		JobName: jobName,
	}
}

func (deployer *K8sDeployer) DeleteTaskFromMonitoring(taskId string) {
	if task, ok := deployer.DeployedTasks[taskId]; ok {
		zap.S().Debugf("Stop monitoring pod of job %s", task.JobName)

		delete(deployer.DeployedTasks, taskId)
	}
}

func (deployer *K8sDeployer) GetMonitoredPodStatuses() (map[string]TaskHealthDetails, error) {
	ctx := context.Background()

	for _, pod := range deployer.DeployedTasks {
		client, err := deployer.actionConfig.KubernetesClientSet()
		if err != nil {
			zap.S().Errorf("Failed to set kubernetes client: %v", err)
			return nil, err
		}

		timeout := int64(podTimeout)
		out, err := client.CoreV1().Pods(deployer.namespace).List(ctx, metav1.ListOptions{
			LabelSelector:  fmt.Sprintf("job-name=%s", pod.JobName),
			TimeoutSeconds: &timeout,
		})
		if err != nil {
			zap.S().Errorf("error getting pods")
			return nil, err
		} else if len(out.Items) == 0 {
			pod.Status = v1.PodStatus{Phase: v1.PodUnknown}
			pod.UnknownCount++
			zap.S().Debugf("Items are empty")
		} else {
			if len(out.Items) > 1 {
				sort.Slice(out.Items, func(i, j int) bool {
					firstPodTime := out.Items[i].ObjectMeta.CreationTimestamp.Time
					secondPodTime := out.Items[j].ObjectMeta.CreationTimestamp.Time
					return firstPodTime.After(secondPodTime)
				})
			}

			lastPod := out.Items[0]

			pod.CreationTime = lastPod.ObjectMeta.CreationTimestamp.Time

			switch lastPod.Status.Phase {
			case v1.PodPending:
				fallthrough

			case v1.PodRunning:
				pod.Status = lastPod.Status
				// reset unknown count
				pod.UnknownCount = 0

			case v1.PodSucceeded:
				deployer.DeleteTaskFromMonitoring(pod.TaskID)

			case v1.PodUnknown:
				pod.UnknownCount++

			case v1.PodFailed:
				fallthrough

			default:
				pod.Status = lastPod.Status
			}
		}

		// update the pod status
		deployer.DeployedTasks[pod.TaskID] = pod
		zap.S().Infof("Pod status of task %s = %s", pod.TaskID, pod.Status.Phase)
	}

	return deployer.DeployedTasks, nil
}
