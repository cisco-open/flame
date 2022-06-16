package main

import "time"

func main() {
	go forever()
	select {} // block forever
}

func forever() {
	for {
		time.Sleep(time.Second)
	}
}
