package util

import (
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/cisco-open/flame/pkg/constants"
)

var fileRegex = regexp.MustCompile(`^(.*/)?(?:$|(.+?)(?:(\.[^.]*$)|$))`)

func SplitPath(path string) (string, string, string) {
	result := fileRegex.FindStringSubmatch(path)

	filePath := result[1]
	fileName := result[2]
	extension := result[3]

	if len(extension) > 0 && strings.Contains(extension, constants.Dot) {
		extension = extension[strings.Index(extension, constants.Dot)+1:]
	}

	return filePath, fileName, extension
}

func MustOpen(f string) *os.File {
	r, err := os.Open(f)
	if err != nil {
		log.Fatalf("Failed to open %s: %v", f, err)
	}

	return r
}
