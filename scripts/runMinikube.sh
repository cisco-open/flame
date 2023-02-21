set -x;

minikube stop

minikube start --driver=hyperkit --cpus=6 --memory=6g --disk-size 100gb --kubernetes-version=v1.23.8 --alsologtostderr -v=7

minikube ssh "echo 'cat << EOF >  /var/lib/boot2docker/bootlocal.sh
echo "DNS=8.8.8.8" >> /etc/systemd/resolved.conf
systemctl restart systemd-resolved
EOF
chmod 755 /var/lib/boot2docker/bootlocal.sh' > commands.sh && chmod +x commands.sh"

minikube ssh "sudo su root < commands.sh"
minikube stop && minikube start
minikube addons enable ingress --alsologtostderr -v=7
minikube addons enable ingress-dns

../fiab/setup-cert-manager.sh

eval $(minikube docker-env)

../fiab/build-image.sh 

afplay /System/Library/Sounds/Glass.aiff

set +x;