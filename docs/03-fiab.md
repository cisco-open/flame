# Flame In A Box (fiab)

## Overview

fiab is a development environment for flame.
The flame system consists of four components: apiserver, controller, notifier and flamelet.
It also includes mongodb as backend state store.
This development environment is mainly tested under MacOS.
This guideline is primarily based on MacOS.
However, this dev environment doesn't work under latest Apple machines with M1 chip set because hyperkit is not yet supported for M1 Mac.
The fiab is also tested under Archlinux. Hence, it may work on other Linux distributions such as Ubuntu.

The `flame/fiab` folder contains several scripts to configure and set up the fiab environment.
Thus, the working directory for this guideline is `flame/fiab`.

## Prerequisites
Install prerequisites for a given distribution as per documents provided below
* [Linux](03.A-fiab-linux.md)
* [Ec2](03.B-fiab-ec2.md)
* [Mac](03.C-fiab-mac.md)

## Starting minikube
Run the following command to start minikube.
```bash
minikube start
```
The default resource allocation is 2 CPU, 4GB memory and 20GB disk.
In order to change these parameters, use `--cpus`, `--memory` and `--disk-size` respectively.
For example,
```bash
minikube start --cpus 4 --memory 4096m --disk-size 100gb
```
When `docker` driver is in use, run ```minikube config set driver docker``` to make docker driver default. These changes will take effect upon a minikube delete and then a minikube start.

We recommend a disk space of 100GB to allow sufficient disk space to store the flame container images and other images in the minikube VM.

Next, `ingress` and `ingress` addons need to be installed with the following command:
```bash
minikube addons enable ingress
minikube addons enable ingress-dns
```
When `hyperkit` driver is in use, enabling `ingress` addon may fail due to the same issue shown in [here](#fixing-docker-build-error),
which explains a workaround. Once the workload is applied, come back here and rerun these commands.


As a final step, a cert manager is needed to enable tls. The `setup-cert-manager.sh` script installs and configures a cert manager for
selfsigned certificate creation. Run the following command:
```bash
./setup-cert-manager.sh
```


## Building flame
A Docker daemon comes within the minikube VM. To build flame container image, set the environment variables with the following command.

```bash
eval $(minikube docker-env)
```
See [here](https://minikube.sigs.k8s.io/docs/handbook/pushing/#1-pushing-directly-to-the-in-cluster-docker-daemon-docker-env) for more details.

To test the config, run the following:
```bash
docker ps
```
This command will show containers within the minikube.

In order to build flame container image, run the following:
```bash
./build-image.sh
```

**Note**: This setup uses docker-daemon within the minikube VM, any downloaded or locally-built images will be gone when the VM is deleted
(i.e., `minikube delete` is executed). Unless a fresh minikube instance is needed, simply stopping the minikube instance would be useful
to save time for development and testing.

To check the flame image built, run `docker images`. An output is similar to:
```bash
REPOSITORY                                TAG       IMAGE ID       CREATED          SIZE
flame                                     latest    e3bf47cdfa66   22 seconds ago   3.96GB
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
When `hyperkit` driver is in use, the `build-image.sh` command may fail with an error similar to the following:
```bash
Get "https://registry-1.docker.io/v2/": dial tcp: lookup registry-1.docker.io on 192.168.64.1:53: read udp 192.168.64.6:48076->192.168.64.1:53: read: connection refused
```
The error might be because of [this issue](https://github.com/kubernetes/minikube/issues/3036).
If minikube is running on a machine with dnscrypt-proxy or dnsmasq, 
refer to [here](https://gist.github.com/rscottwatson/e0e3c890b3d4aa81e46bf2993e3e216f) for more details.
Especially, in case of dnscrypt-proxy, the following workaround is applied.

First, log into the minikube VM with the following command.
```bash
minikube ssh
```
After logging into the VM, become a super user with the following command.
```bash
sudo su -
```
Now run the following:
```bash
cat << EOF >  /var/lib/boot2docker/bootlocal.sh
echo "DNS=8.8.8.8" >> /etc/systemd/resolved.conf 
systemctl restart systemd-resolved
EOF
chmod 755  /var/lib/boot2docker/bootlocal.sh
```

To apply the change, run the following after getting out of the minikube VM:
```bash
minikube stop && minikube start
```
**Note**: restarting the minikube VM may take a while. If the command hangs, press `Ctrl-C` and rerun the command.

## Starting flame
Open a new terminal window and start the minikube tunnel with the following command:
```bash
minikube tunnel
```
The tunnel creates a routable IP for deployment.


To bring up flame and its dependent applications, `helm` is used.
A shell script (`flame.sh`) to use helm is provided.
Run the following command:
```bash
./flame.sh start
```
During the configuration by `flame.sh`, it asks a password for sudo permission.
The reason for this is to add a dns configuration in `/etc/resolver/flame-test`.
When stopping flame, the script asks again a password to delete `/etc/resolver/flame-test`.

The file may look like the following on MacOS:
```
domain flame.test
nameserver 192.168.64.62
search_order 1
timeout 5
```
Here `192.168.64.62` is minikube's IP address.

## Validating deployment
To check deployment status, run the following command:
```bash
kubectl get pods -n flame
```

An example output looks like the following:
```console
NAME                                READY   STATUS    RESTARTS       AGE
flame-apiserver-5df5fb6bc4-22z6l    1/1     Running   0              7m5s
flame-controller-566684676b-g4pwr   1/1     Running   6 (4m4s ago)   7m5s
flame-mlflow-965c86b47-vd8th        1/1     Running   0              7m5s
flame-mongodb-0                     1/1     Running   0              3m41s
flame-mongodb-1                     1/1     Running   0              4m3s
flame-mongodb-arbiter-0             1/1     Running   0              7m5s
flame-mosquitto-6754567c88-rfmk7    1/1     Running   0              7m5s
flame-mosquitto2-676596996b-d5dzj   1/1     Running   0              7m5s
flame-notifier-cf4854cd9-g27wj      1/1     Running   0              7m5s
postgres-7fd96c847c-6qdpv           1/1     Running   0              7m5s
```

As a way to test a successful configuration of routing and dns, test with the following commands:
```bash
ping -c 1 apiserver.flame.test
ping -c 1 notifier.flame.test
ping -c 1 mlflow.flame.test
```
These ping commands should run successfully without any error. As another alternative, open a browser and go to `mlflow.flame.test`.
That should return a mlflow's web page.

## Stopping flame
```bash
./flame.sh stop
```
Before starting flame again, make sure that all the pods in the flame namespace are deleted.
To check that, use `kubectl get pods -n flame` command.

## Logging into a pod
In kubernetes, a pod is the smallest, most basic deployable object. A pod consists of at least one container instance.
Using the pod's name (e.g., `flame-apiserver-65d8c7fcf4-z8x5b`), one can log into the running pod as follows:
```bash
kubectl exec -it -n flame flame-apiserver-65d8c7fcf4-z8x5b -- bash
```

Logs of flame components are found at `/var/log/flame` in the instance.

## Creating flame config
The following command creates `config.yaml` under `$HOME/.flame`.
```bash
./build-config.sh
```
The flame CLI tool, `flamectl` uses the configuration file to interact with the flame system.
In order to build, `flamectl`, run `make install` from the level folder (i.e., `flame`).
This command compiles source code and installs `flamectl` binary as well as other binaries into `$HOME/.flame/bin`.
You may want to add `export PATH="$HOME/.flame/bin:$PATH"` to your shell config (e.g., `~/.zshrc`, `~/.bashrc`) and then restart your terminal.

## Cleanup
To terminate the fiab environment, run the following:
```bash
minikube delete
```

## Running a test ML job
In order to run a sample mnist job, refer to instructions at [mnist example](04-examples.md#mnist).

**Note**: By executing the above command, any downloaded or locally-built images are also deleted together when the VM is deleted.
Unless a fresh minikube instance is needed, simply stopping the minikube (i.e., `minikube stop`) instance would be useful
to save time for development and testing.
