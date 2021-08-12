package objects

import "strconv"

type ServerInfo struct {
	Name string `json:"name,omitempty"`

	IP string `json:"ip,omitempty"`

	Port uint16 `json:"port,omitempty"`

	Uuid string `json:"uuid,omitempty"`

	Tags string `json:"tags,omitempty"`

	Role string `json:"role,omitempty"`

	State string `json:"state,omitempty"`

	//TODO remove me after demo-day
	Command []string `json:"command,omitempty"`

	//required by the controller to check what type of notification to send. Ideally K8 will provide a new node and system will be able to determine it
	IsExistingNode bool `yaml:"is_existing_node" json:"is_existing_node"`

	//required by the controller to check if anything related to the node got updated.
	// Example - schema design change impacted this node so a notification is required to be sent.
	IsUpdated bool `yaml:"is_updated" json:"is_updated"`
}

func (s *ServerInfo) GetAddress() string {
	return s.IP + ":" + strconv.Itoa(int(s.Port))
}
