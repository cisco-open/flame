package job

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	pbNotify "wwwin-github.cisco.com/eti/fledge/pkg/proto/notification"
)

const (
	jobStatusCheckDuration = 1 * time.Minute
)

type handler struct {
	dbService  database.DBService
	jobId      string
	eventQ     *EventQ
	jobQueues  map[string]*EventQ
	mu         *sync.Mutex
	notifierEp string

	jobSpec openapi.JobSpec

	startTime time.Time

	// isDone is set in case where nothing can be done by a handler
	// isDone should be set only once as its buffer size is 1
	isDone chan bool
}

func NewHandler(dbService database.DBService, jobId string, eventQ *EventQ, jobQueues map[string]*EventQ, mu *sync.Mutex,
	notifierEp string) *handler {
	return &handler{
		dbService:  dbService,
		jobId:      jobId,
		eventQ:     eventQ,
		jobQueues:  jobQueues,
		mu:         mu,
		notifierEp: notifierEp,
		isDone:     make(chan bool, 1),
	}
}

func (h *handler) Do() {
	defer func() {
		h.mu.Lock()
		delete(h.jobQueues, h.jobId)
		h.mu.Unlock()
		zap.S().Infof("deleted an eventQ for %s from job queues", h.jobId)
	}()

	jobSpec, err := h.dbService.GetJobById(h.jobId)
	if err != nil {
		zap.S().Errorf("failed to fetch job specification: %v", err)
		return
	}
	h.jobSpec = jobSpec

	h.startTime = time.Now()
	ticker := time.NewTicker(jobStatusCheckDuration)
	for {
		select {
		case <-ticker.C:
			h.checkProgress()

		case event := <-h.eventQ.GetEventBuffer():
			h.doHandle(event)

		case <-h.isDone:
			h.cleanup()
			return
		}
	}
}

// checkProgress checks the progress of a job and returns a boolean flag to tell if the job handling thread can be finished.
// If the job is in completed/failed/terminated state, this method returns true (meaning that the thread can be finished).
// It also returns true if the maximum wait time exceeds. Otherwise, it returns false.
func (h *handler) checkProgress() {
	zap.S().Infof("not yet fully implemented")

	// if the job didn't finish after max wait time, we finish the job handling/monitoring work
	if h.startTime.Add(time.Duration(h.jobSpec.MaxRunTime) * time.Second).Before(time.Now()) {
		zap.S().Infof("Job timed out")

		jobStatus := openapi.JobStatus{State: openapi.FAILED}
		_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, jobStatus)

		h.isDone <- true
		return
	}

	jobStatus, err := h.dbService.GetJobStatus(h.jobSpec.UserId, h.jobId)
	if err != nil {
		zap.S().Warnf("failed to get job status: %v", err)
		return
	}

	if jobStatus.State == openapi.DEPLOYING {
		// TODO: check if at least one node per role is running.
		//       if so, set the job state to RUNNING.
		zap.S().Info("not yet implemented")
	}
}

func (h *handler) doHandle(event *JobEvent) {
	switch event.JobStatus.State {
	// the following six states are managed/updated by controller internally
	// user cannot directly set these states
	case openapi.READY:
		fallthrough
	case openapi.DEPLOYING:
		fallthrough
	case openapi.RUNNING:
		fallthrough
	case openapi.FAILED:
		fallthrough
	case openapi.TERMINATED:
		fallthrough
	case openapi.COMPLETED:
		event.ErrCh <- fmt.Errorf("status update operation not allowed")

	// user is allowed to specify the following three states only:
	case openapi.STARTING:
		h.handleStart(event)

	case openapi.STOPPING:
		h.handleStop(event)

	case openapi.APPLYING:
		h.handleApply(event)

	default:
		event.ErrCh <- fmt.Errorf("unknown state")
	}
}

func (h *handler) handleStart(event *JobEvent) {
	// 1. check the current state of the job
	// 1-1. If the state is not ready / failed / terminated,
	//      return error with a message that a job is not in a scheduleable state
	// 1-2. Otherwise, proceed to the next step

	// 2. Generate task (configuration and ML code) based on the job spec
	// 2-1. If task generation failed, return error
	// 2-2. Otherwise, proceed to the next step

	// 3. update job state to STARTING in the db
	// 3-1. If successful, return nil error to user because it takes time to spin up compute resources
	// 3-2. Otherwise, return error

	// 4. spin up compute resources and handle remaining taaks.
	// 4-1. Once a minimum set of compute resources are provisioned (e.g., one compute node per role),
	//      set the state to RUNNING in the db.
	// 4-2. If the condition in 5-1 is not met for a certain duration, cancel the provisioning of
	//      the compute resources and destroy the provisioned compute resources; set the state to FAILED

	// 5. send start-job event to the agent (i.e., fledgelet) in all the provisioned compute nodes

	zap.S().Infof("requester: %s, jobId: %s", event.Requester, event.JobStatus.Id)

	curStatus, err := h.dbService.GetJobStatus(event.Requester, event.JobStatus.Id)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to check the current status: %v", err)
		return
	}

	if !(curStatus.State == openapi.COMPLETED || curStatus.State == openapi.FAILED ||
		curStatus.State == openapi.TERMINATED || curStatus.State == openapi.READY) {
		event.ErrCh <- fmt.Errorf("job not in a state to schedule")
		return
	}

	// Obtain job specification
	jobSpec, err := h.dbService.GetJob(event.Requester, event.JobStatus.Id)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch job spec")
		return
	}

	tasks, err := newJobBuilder(h.dbService, jobSpec).getTasks()
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to generate tasks: %v", err)
		return
	}

	err = h.dbService.CreateTasks(tasks)
	if err != nil {
		event.ErrCh <- err
		return
	}

	// set the job state to STARTING
	err = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to set job state to %s: %v", event.JobStatus.State, err)

		h.isDone <- true
		return
	}

	// At this point, inform that there is no error by setting the error channel nil
	// This only means that the start-job request was accepted after basic check and
	// the job will be scheduled.
	event.ErrCh <- nil

	// TODO: spin up compute resources

	req := &pbNotify.EventRequest{
		Type:     pbNotify.EventType_START_JOB,
		AgentIds: make([]string, 0),
	}

	for _, task := range tasks {
		req.JobId = task.JobId
		req.AgentIds = append(req.AgentIds, task.AgentId)
	}

	resp, err := newNotifyClient(h.notifierEp).sendNotification(req)
	if err != nil || resp.Status == pbNotify.Response_ERROR {
		event.JobStatus.State = openapi.FAILED
		_ = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)

		h.isDone <- true
		return
	}

	event.JobStatus.State = openapi.DEPLOYING
	_ = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
}

func (h *handler) handleStop(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")
}

func (h *handler) handleApply(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")
}

func (h *handler) cleanup() {
	zap.S().Infof("not yet implemented")

	// TODO: implement step 1
	// 1. decommission compute resources if they are in use

	// 2. wipe out tasks in task DB collection
	err := h.dbService.DeleteTasks(h.jobId)
	if err != nil {
		zap.S().Warnf("failed to delete tasks for job %s: %v", h.jobId, err)
	}
}
