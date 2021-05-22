package main

import (
	"os"

	klog "k8s.io/klog/v2"
	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/cmd"
)

func main() {
	klog.InitFlags(nil)

	defer klog.Flush()

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
