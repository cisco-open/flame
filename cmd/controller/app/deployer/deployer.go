package deployer

import "fmt"

const (
	AKE = "ake" // Microsoft’s Azure Kubernetes Service
	EKS = "eks" // Amazon’s Elastic Kubernetes Service
	GKE = "gke" // Google’s Kubernetes Engine
	K8S = "k8s" // vanilla kubernetes
)

type Deployer interface {
	Initialize(string, string, string) error
	Install(string, string) error
	Uninstall(string) error
	List() error
}

func NewDeployer(platform string) (Deployer, error) {
	// TODO: support other platforms: AKE, EKS, GKE, etc.
	switch platform {
	case K8S:
		return NewK8sDeployer()
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
