# Flame In A Box (fiab)

## Overview

fiab is a development environment for flame.
The flame system consists of four components: apiserver, controller, notifier and flamelet.
It also includes mongodb as backend state store.
This development environment is mainly tested under MacOS.
This guideline is primarily based on MacOS.
However, this dev environment doesn't work under latest Apple machines with M1 chip set.
The fiab is also tested under Archlinux. Hence, it may work on other Linux distributions such as Ubuntu.

The `flame/fiab` folder contains several scripts to configure and set up the fiab environment.
Thus, the working directory for this guideline is `flame/fiab`.

## Prerequisites

### MacOS
fiab relies on `docker desktop`, `minikube`, `kubectl`, `helm` and `docker`.
Note that the docker subscription service agreement has been updated for `docker desktop`.
Hence, `docker desktop` may not be free. Please checkout the [agreement](https://www.docker.com/products/docker-desktop).


#### Step 1
```
brew install --cask docker
```

Instead of `docker desktop`, `hyperkit` can be used as VM driver in case the `docker desktop` is not free under the new agreement.
To use `hyperkit`,
```
brew install hyperkit docker
```
Here we need to install `docker` to make the docker cli tool available.

#### Step 2
Now we install `minikube`, `kubectl` and `helm` as follows.
```
brew install minikube kubectl helm

# optional
brew install --cask robo-3t
```
Note: `robo-3t` is a GUI tool for MongoDB. This tool comes in handy when debugging mongodb-related issues in the flame system.

### Linux
For linux, no VM hypervisor is needed. The following tools are sufficient: `minikube`, `kubectl`, `helm`, and `docker`.
The fiab env was tested under Archlinux in a x86 machine.

## Starting minikube
Run the following command to start minikube.
```
minikube start
```
The default resource allocation is 2 CPU, 4GB memory and 20GB disk.
In order to change these parameters, use `--cpus`, `--memory` and `--disk-size` respectively.
For example, 
```
minikube start --cpus 4 --memory 4096m --disk-size 10000mb
```
We recommend a disk space of 100GB.


## Building flame
A Docker daemon comes within the minikube VM. To build flame container image, set the environment variables with the following command.

```
eval $(minikube docker-env)
```
See [here](https://minikube.sigs.k8s.io/docs/handbook/pushing/#1-pushing-directly-to-the-in-cluster-docker-daemon-docker-env) for more details.

To test the config, run the following:
```
docker ps
```
This command will show containers within the minikube.

In order to build flame container image, run the following:
```
./build-image.sh
```

**Note**: This setup uses docker-daemon within the minikube VM, any downloaded or locally-built images will be gone when the VM is deleted
(i.e., `minikube delete` is executed). Unless a fresh minikube instance is needed, simply stopping the minikube instance would be useful
to save time for development and testing.

To check the flame image built, run `docker images`. An output is similar to:
```
REPOSITORY                                TAG       IMAGE ID       CREATED          SIZE
flame                                    latest    e3bf47cdfa66   22 seconds ago   3.96GB
k8s.gcr.io/kube-apiserver                 v1.22.3   53224b502ea4   7 weeks ago      128MB
k8s.gcr.io/kube-scheduler                 v1.22.3   0aa9c7e31d30   7 weeks ago      52.7MB
k8s.gcr.io/kube-controller-manager        v1.22.3   05c905cef780   7 weeks ago      122MB
k8s.gcr.io/kube-proxy                     v1.22.3   6120bd723dce   7 weeks ago      104MB
kubernetesui/dashboard                    v2.3.1    e1482a24335a   6 months ago     220MB
k8s.gcr.io/etcd                           3.5.0-0   004811815584   6 months ago     295MB
kubernetesui/metrics-scraper              v1.0.7    7801cfc6d5c0   6 months ago     34.4MB
k8s.gcr.io/coredns/coredns                v1.8.4    8d147537fb7d   6 months ago     47.6MB
gcr.io/k8s-minikube/storage-provisioner   v5        6e38f40d628d   8 months ago     31.5MB
k8s.gcr.io/pause                          3.5       ed210e3e4a5b   9 months ago     683kB
```


### Fixing docker build error
The `build-image.sh` command may fail with an error similar to the following:
```
Get "https://registry-1.docker.io/v2/": dial tcp: lookup registry-1.docker.io on 192.168.64.1:53: read udp 192.168.64.6:48076->192.168.64.1:53: read: connection refused
```
The error might be because of [this issue](https://github.com/kubernetes/minikube/issues/3036).
If minikube is running on a machine with dnscrypt-proxy or dnsmasq, 
refer to [here](https://gist.github.com/rscottwatson/e0e3c890b3d4aa81e46bf2993e3e216f) for more details.
Especially, in case of dnscrypt-proxy, the following workaround is applied.

First, log into the minikube vm with the following command.
```
minikube ssh
```
After logging into the vm, become a super user with the following command.
```
sudo su -
```
Now run the following:
```
cat << EOF >  /var/lib/boot2docker/bootlocal.sh
echo "DNS=8.8.8.8" >> /etc/systemd/resolved.conf 
systemctl restart systemd-resolved
EOF
chmod 755  /var/lib/boot2docker/bootlocal.sh
```

To apply the change, run the following after getting out of the minikube vm:
```
minikube stop && minikube start
```
**Note**: restarting the minikube vm may take a while. If the command hangs, press `Ctrl-C` and rerun the command.

## Starting flame
Open a new terminal window and start the minikube tunnel with the following command:
```
minikube tunnel
```
The tunnel creates a routable IP for deployment.


To bring up flame and its dependent applications, `helm` is used.
A shell script to use helm is provided
Run the following command:
```
./flame.sh start
```

To check deployment status, run the following command:
```
kubectl get pods -n flame
```

An example output looks like the following:
```
NAME                                READY   STATUS    RESTARTS       AGE
flame-apiserver-65d8c7fcf4-kn6ck   1/1     Running   0              2m16s
flame-controller-f6c99d8d5-5jfd9   1/1     Running   3 (112s ago)   2m16s
flame-mongodb-0                    1/1     Running   0              49s
flame-mongodb-1                    1/1     Running   0              71s
flame-mongodb-arbiter-0            1/1     Running   0              2m16s
flame-notifier-c59bbcf65-qqvrc     1/1     Running   0              2m16s
mlflow-6dd895c889-42nfl             1/1     Running   0              2m16s
postgres-748c47694c-94lhh           1/1     Running   0              2m16s
```

## Stopping flame
```
./flame.sh stop
```
Before starting flame again, make sure that all the pods in the flame namespace are deleted.
To check that, use `kubectl get pods -n flame` command.

## Logging into a pod
In kubernetes, a pod is the smallest, most basic deployable object. A pod consists of at least one container instance.
Using the pod's name (e.g., `flame-apiserver-65d8c7fcf4-z8x5b`), one can log into the running pod as follows:
```
kubectl exec -it -n flame flame-apiserver-65d8c7fcf4-z8x5b -- bash
```

Logs of flame components are found at `/var/log/flame` in the instance.

## Creating flame config
The following command creates `config.yaml` under `$HOME/.flame`.
```
./build-config.sh
```
The flame CLI tool, `flamectl` uses the configuration file to interact with the flame system.
In order to build, `flamectl`, run `make install` from the level folder (i.e., `flame`).
This command compiles source code and installs `flamectl` binary as well as other binaries into `$HOME/.flame/bin`.

## Cleanup
To terminate the fiab environment, run the following:
```
minikube delete
```

## Running a test ML job
In order to run a sample mnist job, refer to instructions at [mnist example](04-examples.md#mnist).

**Note**: By executing the above command, any downloaded or locally-built images are also deleted togehter when the VM is deleted.
Unless a fresh minikube instance is needed, simply stopping the minikube (i.e., `minikube stop`) instance would be useful
to save time for development and testing.
