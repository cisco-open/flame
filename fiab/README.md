# fledge in a box (fiab)

fiab is a development environment for fledge. The fledge system consists of four components: apiserver, controller, notifier and fledgelet.
It also includes mongodb as backend state store. The fiab mainly relies on docker and docker-compose. This development environment is mainly
tested under MacOS. This guideline will be primarily based on MacOS. However, the fiab may work in a linux machine.

## Prerequisites
To run fiab, two software tools are needed: `docker` and `docker-compose`.

The installation instructions are based on `brew` on MacOS.
```
brew install --cask docker
brew install docker-compose
```

Docker should be running too. All the `docker-compose` command MUST be executed in the directory (i.e., `fiab`)
where `docker-compose.yaml` file is.

## Building fledge
In order to build fledge system, run the following:
```
docker-compose build
```

## Running fledge
```
docker-compose up
```

To run fledge in a detached mode, run `docker-compose up -d`

Example output:
```
$ docker-compose up -d
Creating network "fiab_default" with the default driver
Creating fledge-notifier ... done
Creating fledge-db       ... done
Creating fledge-controller ... done
Creating fledge-apiserver  ... done
```

# Checking
To check if all services are up and running, run the following:
```
$ docker-compose ps
      Name                     Command               State                                                                    Ports
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fledge-apiserver    /fledge/apiserver start -- ...   Up      0.0.0.0:10100->10100/tcp,:::10100->10100/tcp
fledge-controller   bash -c sleep 5 && /fledge ...   Up      0.0.0.0:10102->10102/tcp,:::10102->10102/tcp
fledge-db           docker-entrypoint.sh mongod      Up      0.0.0.0:27017->27017/tcp,:::27017->27017/tcp, 0.0.0.0:27018->27018/tcp,:::27018->27018/tcp, 0.0.0.0:27019->27019/tcp,:::27019->27019/tcp
fledge-notifier     /fledge/notifier                 Up      0.0.0.0:10101->10101/tcp,:::10101->10101/tcp
```

## Logging into container
Using the container's name, one can log into a running container as follows:
```
docker exec -it fledge-apiserver bash
```

Logs of fledge components are found at `/var/log/fledge` within a container.

## Cleanup
To terminate the fiab environment, run the following:
```
docker-compose down
```

Example:
```
$ docker-compose down
Stopping fledge-apiserver  ... done
Stopping fledge-controller ... done
Stopping fledge-db         ... done
Stopping fledge-notifier   ... done
Removing fledge-apiserver  ... done
Removing fledge-controller ... done
Removing fledge-db         ... done
Removing fledge-notifier   ... done
Removing network fiab_default
```

If `docker-compose` is executed without `-d` flag, then press `ctrl-c` and run `docker-compose rm`.


## Vagrant Environment (Deprecated)
The vagrant environment based on virtualbox is now deprecated. 
