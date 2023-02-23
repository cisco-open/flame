package dataset

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type IDataset interface {
	Create(params *models.DatasetParams) error
	Get(params *models.DatasetParams) error
	GetMany(params *models.DatasetParams, flagAll bool) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) IDataset {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
