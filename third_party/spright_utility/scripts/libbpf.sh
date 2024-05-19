#/bin/bash
# This script can be run with non-root user

echo "Installing libbpf"

git clone --single-branch https://github.com/libbpf/libbpf.git
cd libbpf
git switch --detach v0.6.0
cd src
make -j $(nproc)
sudo make install
echo "/usr/lib64/" | sudo tee -a /etc/ld.so.conf
sudo ldconfig
cd ../..