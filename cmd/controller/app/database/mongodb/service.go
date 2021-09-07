package mongodb

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
	"go.mongodb.org/mongo-driver/x/bsonx"
	"go.uber.org/zap"
)

const (
	DatabaseName     = "fledge"
	DesignCollection = "t_design"
	JobCollection    = "t_job"

	orderAscend  int32 = 1
	orderDescend int32 = -1
)

type MongoService struct {
	client           *mongo.Client
	database         *mongo.Database
	designCollection *mongo.Collection
	jobCollection    *mongo.Collection
}

var mongoDB = &MongoService{}

func NewMongoService(uri string) (*MongoService, error) {
	// create the base context, the context in which your application runs
	//https://www.mongodb.com/blog/post/quick-start-golang-mongodb-starting-and-setup
	//10seconds = timeout duration that we want to use when trying to connect.
	//ctx, cancel := context.WithCancel(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	client, err := mongo.NewClient(options.Client().ApplyURI(uri))
	if err != nil {
		zap.S().Fatalf("error creating mongo client: %+v", err)
		panic(err)
	}
	// add disconnect call here
	//todo can't call defer here because once method returns db connection closes. When to close the connection?
	//https://codereview.stackexchange.com/questions/241739/connection-to-mongodb-in-golang
	//defer func() {
	//	zap.S().Infof("closing database connection ...")
	//	if err = client.Disconnect(ctx); err != nil {
	//		panic(err)
	//	}
	//}()

	if err := client.Connect(ctx); err != nil {
		zap.S().Fatalf("failed to connect to MongoDB: %+v", err)
		panic(err)
	}

	// Ping the primary to check connection establishment
	if err := client.Ping(ctx, readpref.Primary()); err != nil {
		panic(err)
	}
	zap.S().Infof("Successfully connected to database and pinged")

	db := client.Database(DatabaseName)
	mongoDB = &MongoService{
		client:           client,
		database:         db,
		designCollection: db.Collection(DesignCollection),
		jobCollection:    db.Collection(JobCollection),
	}

	kv := map[string]int32{"userid": orderAscend, "id": orderAscend}
	if err := createUniqueIndex(mongoDB.designCollection, ctx, kv); err != nil {
		return nil, err
	}

	return mongoDB, nil
}

func createUniqueIndex(coll *mongo.Collection, ctx context.Context, kv map[string]int32) error {
	keysDoc := bsonx.Doc{}

	for key, val := range kv {
		if val != orderAscend && val != orderDescend {
			val = orderAscend
		}
		keysDoc = keysDoc.Append(key, bsonx.Int32(val))
	}

	model := mongo.IndexModel{
		Keys: keysDoc,
		// create UniqueIndex option
		Options: options.Index().SetUnique(true),
	}

	index, err := coll.Indexes().CreateOne(ctx, model)
	if err != nil {
		zap.S().Debugf("Failed to created index for %s: %v", coll.Name(), err)
		return err
	}

	zap.S().Infof("Created index %s for %s successfully", index, coll.Name())

	return nil
}
