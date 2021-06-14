#!/usr/bin/env bash

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
DEFAULT='\033[39m'

# - - - - - - - - - - - - - - - - - - - - -
# Packages
# - - - - - - - - - - - - - - - - - - - - -
echo -e $YELLOW"Checking import installation..."
`go mod tidy`
echo -e $GREEN"Done!\n"

# - - - - - - - - - - - - - - - - - - - - -
# Building Binary
# - - - - - - - - - - - - - - - - - - - - -
binPath="../../bin"
executable="apiServer"
mainFile="apiserver.go"

echo -e $YELLOW"Building and installing...."
echo -e $DEFAULT"Installation Path  = "$binPath
echo "Build Filename      = "$mainFile
echo "Executable Filename = "$executable
 
go build  -o $binPath"/"$executable $mainFile

echo -e $GREEN"Done!"
