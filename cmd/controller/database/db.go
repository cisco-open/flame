package database

import (
	"strings"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database/mongodb"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var DB StoreCollection

func NewDBService(uri string) {
	dbName := strings.Split(uri, ":")[0]
	switch dbName {
	case util.MONGODB:
		DB, _ = mongodb.NewMongoService(uri)
		// TODO: add error handling
	case util.MySQL:
		fallthrough
	default:
		zap.S().Fatalf("Unknown DB type: %s", dbName)
	}
}
