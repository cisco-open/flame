package task

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type ITask interface {
	Get(params *models.TaskParams) error
	GetMany(params *models.TaskParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) ITask {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
