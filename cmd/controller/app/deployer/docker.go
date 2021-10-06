package deployer

import (
	"go.uber.org/zap"
)

// NOTE: DockerDeployer doesn't support dynamic deployment as docker and
//       docker-compose are only for local development purpose
type DockerDeployer struct{}

func NewDockerDeployer() (*DockerDeployer, error) {
	return &DockerDeployer{}, nil
}

func (deployer *DockerDeployer) Initialize(_ string, _ string) error {
	zap.S().Infof("not supported")

	return nil
}

func (deployer *DockerDeployer) Install(_ string, _ string) error {
	zap.S().Infof("not supported")

	return nil
}

func (deployer *DockerDeployer) Uninstall(_ string) error {
	zap.S().Infof("not supported")

	return nil
}

func (deployer *DockerDeployer) List() error {
	zap.S().Infof("not supported")

	return nil
}
