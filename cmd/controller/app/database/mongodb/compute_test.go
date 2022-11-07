package mongodb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/integration/mtest"

	"github.com/cisco-open/flame/pkg/openapi"
)

func TestMongoService_UpdateDeploymentStatus(t *testing.T) {
	mt := mtest.New(t, mtest.NewOptions().ClientType(mtest.Mock))
	defer mt.Close()
	mt.Run("success", func(mt *mtest.T) {
		db := &MongoService{
			deploymentCollection: mt.Coll,
		}
		mt.AddMockResponses(mtest.CreateSuccessResponse(), mtest.CreateCursorResponse(1, "flame.deployment", mtest.FirstBatch, bson.D{}))
		err := db.UpdateDeploymentStatus("test jobid","test compute id",
			map[string]openapi.AgentState{"test task id": openapi.AGENT_DEPLOY_SUCCESS,
		})
		assert.Nil(t, err)
	})
	mt.Run("status update failure", func(mt *mtest.T) {
		db := &MongoService{
			deploymentCollection: mt.Coll,
		}
		mt.AddMockResponses(
			mtest.CreateCursorResponse(1, "flame.deployment", mtest.FirstBatch, bson.D{{"_id", "1"}}))
		err := db.UpdateDeploymentStatus("test jobid", "test compute id",
			map[string]openapi.AgentState{"test task id": openapi.AGENT_DEPLOY_SUCCESS},
		)
		assert.NotNil(t, err)
	})
}
