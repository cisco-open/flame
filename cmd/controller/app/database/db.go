package database

import (
	"fmt"
	"strings"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database/mongodb"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var DB StoreCollection

func NewDBService(uri string) error {
	dbName := strings.Split(uri, ":")[0]

	var err error

	switch dbName {
	case util.MONGODB:
		DB, err = mongodb.NewMongoService(uri)

	case util.MySQL:
		fallthrough

	default:
		err = fmt.Errorf("unknown DB type: %s", dbName)
	}

	return err
}
