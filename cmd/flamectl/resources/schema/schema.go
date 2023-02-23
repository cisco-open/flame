package schema

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type ISchema interface {
	Create(params *models.SchemaParams) error
	Get(params *models.SchemaParams) error
	GetMany(params *models.SchemaParams) error
	Update(params *models.SchemaParams) error
	Remove(params *models.SchemaParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) ISchema {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
