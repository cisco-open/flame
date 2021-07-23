package objects

import "strconv"

type ServerInfo struct {
	Name string `json:"name,omitempty"`

	IP string `json:"ip,omitempty"`

	Port uint16 `json:"port,omitempty"`

	Uuid string `json:"uuid,omitempty"`

	Tags string `json:"tags,omitempty"`

	Role string `json:"role,omitempty"`

	Status string `json:"status,omitempty"`
}

func (s *ServerInfo) GetAddress() string {
	return s.IP + ":" + strconv.Itoa(int(s.Port))
}
