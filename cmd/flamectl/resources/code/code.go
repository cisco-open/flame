package code

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/samber/do"
	"go.uber.org/zap"
)

type ICode interface {
	Create(params *models.CodeParams) error
	Get(params *models.CodeParams) error
	Remove(params *models.CodeParams) error
}

type container struct {
	logger *zap.Logger
}

// New initializes a new instance of the container struct with the specified config
func New(i *do.Injector) ICode {
	return &container{
		logger: do.MustInvoke[*zap.Logger](i),
	}
}
