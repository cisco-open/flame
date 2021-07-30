package app

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"os/exec"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

func NewJobInitApp(info objects.AppConf) {
	job := info.JobInfo
	zap.S().Debugf("Init new job. Job id: %s | design id: %s | schema id: %s | role: %s", job.ID, job.DesignId, job.SchemaId, info.Role)

	//Step 1 : dump the information in application confirmation json file
	err := initApp(info)
	if err != nil {
		zap.S().Errorf("error intializing configuration details for application")
	} else {
		// TODO currently hard coded to run the simple example. Work to be done includes - making it generic and putting in into a go routine.
		//Step 2: start the application
		command := ""
		if info.Role == "foo" {
			command = "/fledge/example/simple/foo/main.py"
		} else if info.Role == "bar" {
			command = "/fledge/example/simple/bar/main.py"
		}
		cmd := exec.Command("python3", command)
		outfile, err := os.Create("./out.txt")
		if err != nil {
			panic(err)
		}
		cmd.Stdout = outfile
		//cmd.Stdout = os.Stdout
		err = cmd.Start()
		if err != nil {
			zap.S().Errorf("error starting the application. %v", err)
		}
	}
}

func initApp(info objects.AppConf) error {
	file, _ := json.MarshalIndent(info, "", " ")
	//directory path
	filepath := "/fledge/job/" + info.JobInfo.ID
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		os.MkdirAll(filepath, 0700) // Create your file
	}
	confFilePath := filepath + "/conf.json"
	//create a configuration file for the application
	err := ioutil.WriteFile(confFilePath, file, 0644)
	return err
}
