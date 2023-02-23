package job

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type IJob interface {
	Create(params *models.JobParams) error
	Get(params *models.JobParams) error
	GetMany(params *models.JobParams) error
	Update(params *models.JobParams) error
	Remove(params *models.JobParams) error
	Start(params *models.JobParams) error
	Stop(params *models.JobParams) error
	GetStatus(params *models.JobParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) IJob {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
