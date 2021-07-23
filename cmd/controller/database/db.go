package database

import (
	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database/mongodb"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

var DB StoreCollection

type DBInfo struct {
	Name string
	URI  string
}

func NewDBService(dbObj DBInfo) {
	switch db := dbObj.Name; db {
	case util.MONGODB:
		DB, _ = mongodb.NewMongoService(dbObj.URI)
		//	todo add error handling
	//case util2.MySQL:
	//	//	placeholder implementation
	//	DB, _ = mysql2.NewMySqlService()
	default:
		zap.S().Fatalf("Please pass DB Type - supports: Mongodb")
	}
}
