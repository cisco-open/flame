#/bin/bash
# This script can be run with non-root user

sudo apt update && sudo apt install -y flex bison build-essential dwarves libssl-dev libelf-dev \
                    libnuma-dev pkg-config python3-pip python3-pyelftools \
                    libconfig-dev golang clang gcc-multilib uuid-dev sysstat

wget --no-hsts https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.16.tar.xz
tar -xf linux-5.16.tar.xz
cd linux-5.16
make olddefconfig
scripts/config --set-str SYSTEM_TRUSTED_KEYS ""
scripts/config --set-str SYSTEM_REVOCATION_KEYS ""
make -j $(nproc)
find -name *.ko -exec strip --strip-unneeded {} +
sudo make modules_install -j $(nproc)
sudo make install
cd ..

sudo reboot