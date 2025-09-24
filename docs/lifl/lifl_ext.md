# Running LIFL on a Mounted Drive for Extra Storage

This guide will walk you through configuring and running LIFL on a mounted drive, such as /mydata, for extra storage. You can name the mounted drive as needed, just replace /mydata with your desired mount point.

## Prerequisites

Before starting, ensure you have:

- A Linux-based system
- A mounted drive (e.g., /mydata)
- Sufficient storage space for datasets, dependencies, and binaries.

> [!NOTE]
> In this example the mounted drive is named `/mydata`

### 1. Prepare the Mounted Drive

First, update your system and install essential utilities.

```bash
sudo apt update && sudo apt install -y byobu htop
```

Change ownership of the mounted drive to the current user:

```bash
sudo chown -R $(id -u):$(id -g) /mydata
cd /mydata
export MYMOUNT=/mydata
```

### 2. Install Golang on the Mounted Drive

Download and extract the latest version of Golang:

```bash
golang_file=go1.22.3.linux-amd64.tar.gz
curl -LO https://go.dev/dl/$golang_file && tar -C /mydata -xzf $golang_file
```

Update the system's PATH:

```bash
echo "PATH=\"/mydata/go/bin:\$PATH\"" >> $HOME/.bashrc
echo "PATH=\"\$HOME/.flame/bin:\$PATH\"" >> $HOME/.bashrc
source $HOME/.bashrc
```

### 3. Install GolangCI-Lint

Install GolangCI-Lint to ensure code quality for Go projects:

```bash
curl -sSfL <https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh> | sh -s -- -b /mydata/go/bin v1.49.0
golangci-lint --version
```

### 4. Install Miniconda on the Mounted Drive

Download and install Miniconda on the mounted drive:

```bash
wget <https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh>
chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -p /mydata/miniconda3
source /mydata/miniconda3/bin/activate
conda init bash
source $HOME/.bashrc
```

### 5. Set up the Python Environment

Create a Python environment for LIFL:

```bash
conda create -n flame python=3.9 -y
conda activate flame
```

Install necessary Python libraries:

```bash
pip install google tensorflow torch torchvision
```

### 6. Clone and Setup the Flame Repository

Clone the Flame repository:

```bash
git clone <https://github.com/cisco-open/flame.git>
```

### 7. Upgrade Kernel (if needed)

Note: If your kernel version is >= 5.15, you can skip this step. To check use the following command `uname -r`

Navigate to the spright_utility scripts directory and execute the kernel upgrade script:

```bash
cd flame/third_party/spright_utility/scripts
./upgrade_kernel.sh
```

### 8. Install libbpf for LIFL

Install the dependencies for libbpf:

```bash
sudo apt update && sudo apt install -y flex bison build-essential dwarves \
        libssl-dev libelf-dev pkg-config libconfig-dev clang gcc-multilib byobu htop
```

Then, install libbpf:

```bash
cd flame/third_party/spright_utility/scripts
./libbpf.sh
```

Finally, install sockmap_manager files

```bash
cd flame/third_party/spright_utility/
make all
```

### 9. Set up a Local MQTT Broker

LIFL uses an MQTT broker for communication during federated learning. To install and set up a local MQTT broker:

```bash
sudo apt update
sudo apt install -y mosquitto
sudo systemctl status mosquitto
```

You should see something like:

```bash
mosquitto.service - Mosquitto MQTT v3.1/v3.1.1 Broker
     Active: active (running)
```

To manage the Mosquitto service:

```bash
# Start Mosquitto

sudo systemctl start mosquitto

# Stop Mosquitto

sudo systemctl stop mosquitto

# Restart Mosquitto

sudo systemctl restart mosquitto
```

Tutorial on how to run an example can be found in [lifl example](./lifl.md#running-an-example)
