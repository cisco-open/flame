export GO111MODULE=on
export GOPRIVATE="wwwin-github.cisco.com"
export GONOPROXY="github.com,gopkg.in,go.uber.org"
export GOPROXY=https://${ARTIFACTORY_USER}:${ARTIFACTORY_PASSWORD}@engci-maven-master.cisco.com/artifactory/api/go/nyota-go

go get ./...
golangci-lint run -v
