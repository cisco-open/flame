package mongodb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/integration/mtest"
)

func TestMongoService_DeleteDesignCode(t *testing.T) {
	mt := mtest.New(t, mtest.NewOptions().ClientType(mtest.Mock))
	defer mt.Close()
	mt.Run("success", func(mt *mtest.T) {
		db := &MongoService{
			designCollection: mt.Coll,
		}
		mt.AddMockResponses(bson.D{{"ok", 1}, {"acknowledged", true},
			{"n", 1}, {"nModified", 1}})
		err := db.DeleteDesignCode("userid", "designid", "version")
		assert.Nil(t, err)
	})

	mt.Run("no document deleted", func(mt *mtest.T) {
		db := &MongoService{
			designCollection: mt.Coll,
		}
		mt.AddMockResponses(bson.D{{"ok", 1}, {"acknowledged", true},
			{"n", 0}, {"nModified", 0}})
		err := db.DeleteDesignCode("userid", "designid", "version")
		assert.NotNil(t, err)
	})
}
