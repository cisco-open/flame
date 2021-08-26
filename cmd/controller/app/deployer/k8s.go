package deployer

import (
	"os"

	"go.uber.org/zap"
	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/kube"
)

type K8sDeployer struct {
	clusterName    string
	namespace      string
	kubeconfigPath string

	actionConfig action.Configuration
}

func NewK8sDeployer() (*K8sDeployer, error) {
	return &K8sDeployer{}, nil
}

func (deployer *K8sDeployer) Initialize(clusterName string, namespace string, kubeconfigPath string) error {
	deployer.clusterName = clusterName
	deployer.namespace = namespace
	deployer.kubeconfigPath = kubeconfigPath

	actionConfig := new(action.Configuration)
	config := kube.GetConfig(deployer.kubeconfigPath, "", deployer.namespace)

	logger := func(format string, v ...interface{}) {
		zap.S().Debugf(format, v)
	}

	err := actionConfig.Init(config, deployer.namespace, os.Getenv("HELM_DRIVER"), logger)
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
