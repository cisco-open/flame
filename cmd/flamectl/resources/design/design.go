package design

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type IDesign interface {
	Create(params *models.DesignParams) error
	Get(params *models.DesignParams) error
	GetMany(params *models.DesignParams) error
	Remove(params *models.DesignParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) IDesign {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
