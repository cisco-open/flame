# Fiab installation in Amazon Linux 2
This guideline is for configuring fiab in amazon linux 2 with GPU supported instance types (e.g., p2).

## Prerequisites
This section is specifically for AWS EC2 instance with amazon linux 2 image and GPU.
For other linux distributions without GPU (regardless of VM or baremetal machine), refer to [Ubuntu](03-a-ubuntu.md);
with their respective package manager, the guideline for Ubuntu can be easily followed.

For Amazon linux 2 image (amzn2), the following tools are necessary: `minikube`, `kubectl`, `helm`, `cri-dockerd`, `crictl` , `docker` and `jq`.
The image was tested under an ec2 instance with GPU (e.g., p2 instances).

To set up fiab, run `install.sh` under the fiab folder.
```bash
cd fiab
./install.sh amzn2
```
**Note: If install.sh is executed, the below steps in the prerequisites section must be skipped.
Go to the [staring minikube part](#Starting-minikube).**

This prerequisites part should be executed only once.
The following shows the steps incorporated in the `install.sh` script,
which can be manually followed to understand what the script does.

### Step 1: Install docker
Install docker as per [this](https://docs.docker.com/engine/install/) document.

### Step 2: Install Docker CRI
``` bash
# set up golang compilation env
wget https://storage.googleapis.com/golang/getgo/installer_linux
chmod +x ./installer_linux
./installer_linux
source ~/.bash_profile

# download cri-docker
git clone https://github.com/Mirantis/cri-dockerd.git
cd cri-dockerd
mkdir bin
go build -o bin/cri-dockerd

# install cri-docker
sudo install -o root -g root -m 0755 bin/cri-dockerd /usr/bin/cri-dockerd
sudo cp -a packaging/systemd/* /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable cri-docker.service
sudo systemctl enable --now cri-docker.socket
```

### Step 3:Install crictl
```bash
# install crictl
VERSION="v1.25.0"
wget https://github.com/kubernetes-sigs/cri-tools/releases/download/$VERSION/crictl-$VERSION-linux-amd64.tar.gz
sudo tar zxvf crictl-$VERSION-linux-amd64.tar.gz -C /usr/local/bin
rm -f crictl-$VERSION-linux-amd64.tar.gz
```

### Step 4: Installing minikube, kubectl and helm
```bash
# install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-latest.x86_64.rpm
sudo rpm -Uvh minikube-latest.x86_64.rpm

# install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# install helm
HELM_VERSION=v3.10.2
curl -LO https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz
tar -zxvf helm-$HELM_VERSION-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm
```

## Starting minikube
A minikube environment is resuable until it is deleted by executing `minikube delete`.
If the minikube env is destroyed, this step needs to be executed.
If it is stopped by running `sudo minikube stop`, one can simply restart it by running `sudo minikube start`
without need to follow the steps below.

### Step 1: Start minikube with none driver
```bash
sudo minikube start --driver=none --apiserver-ips 127.0.0.1 --apiserver-name localhost --cni=bridge
```

Note: If `Exiting due to HOST_JUJU_LOCK_PERMISSION` error happens, run the following command:

```bash
sudo sysctl fs.protected_regular=0
```

Run the following commands to ensure that `kubectl` can be executed without `sudo`:
```bash
# remove any old config
rm -rf $HOME/.kube $HOME/.minikube

# transfer config to a normal user so that kubectl commands can be executed without sudo
sudo cp -rf /root/.kube /root/.minikube $HOME
sudo chown -R $USER $HOME/.kube $HOME/.minikube

# update the cert file's location correctly
sed -i 's@/root@'"$HOME"'@' $HOME/.kube/config
```

### Step 2: Install NVIDIA'S device plugin
1. If NVIDIA's GPU is available in the machine, run the following command to install nvidia device plugin:
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
```

2. To check if GPUs are enabled, run the following command:
```bash
kubectl get nodes -ojson | jq .items[].status.capacity
```
An output should look similar to:
```console
{ 
  "cpu": "4",
  "ephemeral-storage": "524275692Ki",
  "hugepages-1Gi": "0",
  "hugepages-2Mi": "0",
  "memory": "62766704Ki",
  "nvidia.com/gpu": "1",
  "pods": "110"
}
```

### Step 3: Install NVIDIA'S GPU feature discovery resources
More details are found [here](https://github.com/NVIDIA/gpu-feature-discovery).

Deploy Node Feature Discovery (NFD) as a daemonset.
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.7.0/deployments/static/nfd.yaml
```

Deploy NVIDIA GPU Feature Discovery (GFD) as a daemonset.
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-feature-discovery/v0.7.0/deployments/static/gpu-feature-discovery-daemonset.yaml
```

```bash
kubectl get nodes -o yaml
```
The above command will output something similar to the following:
```console
apiVersion: v1
items:
- apiVersion: v1
  kind: Node
  metadata:
    ...
    labels:
      ...
      nvidia.com/cuda.driver.major: "470"
      nvidia.com/cuda.driver.minor: "57"
      nvidia.com/cuda.driver.rev: "02"
      nvidia.com/cuda.runtime.major: "11"
      nvidia.com/cuda.runtime.minor: "4"
      nvidia.com/gfd.timestamp: "1672792567"
      nvidia.com/gpu.compute.major: "3"
      nvidia.com/gpu.compute.minor: "7"
      nvidia.com/gpu.count: "1"
      nvidia.com/gpu.family: kepler
      nvidia.com/gpu.machine: HVM-domU
      nvidia.com/gpu.memory: "11441"
      nvidia.com/gpu.product: Tesla-K80
      nvidia.com/gpu.replicas: "1"
      nvidia.com/mig.capable: "false"
      ...
...
```

### Step 4: Configuring addons
Next, `ingress` and `ingress-dns` addons need to be installed with the following command:
```bash
sudo minikube addons enable ingress
sudo minikube addons enable ingress-dns
```

As a final step, a cert manager is needed to enable tls. The `setup-cert-manager.sh` script installs and configures a cert manager for
selfsigned certificate creation. Run the following command:
```bash
./setup-cert-manager.sh
```

## (Optional) Building flame container image
To simply use flame, skip this step and go to the [starting flame step](#Starting-flame).
Building flame container image is only needed if a developer makes changes in the source code and wants to test the local changes.

In order to build flame container image, run the following:
```bash
./build-image.sh
```

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

## Starting flame
Open a new terminal window and start the minikube tunnel with the following command:
```bash
sudo minikube tunnel
```
The tunnel creates a routable IP for deployment.

To bring up flame and its dependent applications, `helm` is used.
A shell script (`flame.sh`) to use helm is provided.
Run the following command:
```bash
./flame.sh start
```
The above command ensures that the latest official flame image from docker hub is used.
To use a locally developed image, add `--local-img ` in the above command.

**Note**: The following error may occur during the start.
```console
Error: INSTALLATION FAILED: failed post-install: timed out waiting for the condition
```
This issue may be because container images are large or the Internet connection is slow.
The issue has been reported in minikube [github](https://github.com/kubernetes/minikube/issues/14789).
The latest minikube still doesn't contain the patched component (cri-dockerd 0.2.6).
A workaround is to pull images manually (e.g. `minikube ssh docker pull ciscoresearch/flame:latest`).
The command `kubectl get pods -n flame` gives a list of pods and their status.
The pods with `ErrImagePull` or `ImagePullBackOff` status are ones that might be affected by the issue.
Identifying the required image can be done by running a `kubectl describe` command
(e.g., `kubectl describe pod -n flame flame-apiserver-5df5fb6bc4-22z6l`);
the command's output will show details about the pod, including image name and its tag.

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

In amazon ec2, `flame.test` domain needs to be added to Route 53 with the minikube IP address,
which can be obtained by running `minikube ip`. Without route 53 configuration, the following
ping test will fail.

As a way to test a successful configuration of routing and dns, test with the following commands:
```bash
ping -c 1 apiserver.flame.test
ping -c 1 notifier.flame.test
ping -c 1 mlflow.flame.test
```
These ping commands should run successfully without any error. As another alternative, open a browser and go to `mlflow.flame.test`.
That should return a mlflow's web page.

## Stopping flame
Once using flame is done, one can stop flame by running the following command:
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

## Building flamectl
The flame CLI tool, `flamectl` uses the configuration file (`config.yaml`) to interact with the flame system.
In order to build `flamectl`, run `make install` from the level folder (i.e., `flame`).
This command compiles source code and installs `flamectl` binary as well as other binaries into `$HOME/.flame/bin`.
You may want to add `export PATH="$HOME/.flame/bin:$PATH"` to your shell config (e.g., `~/.zshrc`, `~/.bashrc`) and then reload your shell config (e.g., `source ~/.bashrc`).
The examples in [here](04-examples.md) assume that `flamectl` is in `$HOME/.flame/bin` and the path (`$HOME/.flame/bin`) is exported.

## Cleanup
To terminate the fiab environment, run the following:
```bash
sudo minikube delete
```

**Note**: By executing the above command, any downloaded or locally-built images are also deleted together when the VM is deleted.
Unless a fresh minikube instance is needed, simply stopping the minikube (i.e., `sudo minikube stop`) instance would be useful
to save time for development and testing.

## Running a test ML job
In order to run a sample mnist job, refer to instructions at [mnist example](04-examples.md#mnist).
