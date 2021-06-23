package database

import (
	"go.uber.org/zap"
	"wwwin-github.cisco.com/fledge/fledge/cmd/controller/database/mongodb"
	util2 "wwwin-github.cisco.com/fledge/fledge/pkg/util"
)

var DB StoreCollection

type DBInfo struct {
	Name string
	URI  string
}

func NewDBService(dbObj DBInfo) {
	switch db := dbObj.Name; db {
	case util2.MONGODB:
		DB, _ = mongodb.NewMongoService(dbObj.URI)
		//	todo add error handling
	//case util2.MySQL:
	//	//	placeholder implementation
	//	DB, _ = mysql2.NewMySqlService()
	default:
		zap.S().Fatalf("Please pass DB Type - supports: Mongodb")
	}
}
