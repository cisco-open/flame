echo "Running: minikube stop"
minikube stop

echo "Running: minikube delete"
minikube delete

echo "Running: minikube start --cpus=6 --memory=6g --disk-size 100gb"
minikube start --cpus=6 --memory=6g --disk-size 100gb

echo "Running: minikube ssh \"sudo systemctl stop systemd-resolved\""
minikube ssh "sudo systemctl stop systemd-resolved"

echo "Running: minikube ssh \"sudo systemctl disable systemd-resolved\""
minikube ssh "sudo systemctl disable systemd-resolved"

echo "Running: minikube ssh \"printf 'nameserver 8.8.8.8\nsearch .\n' | sudo tee /etc/resolv.conf\""
minikube ssh "printf 'nameserver 8.8.8.8\nsearch .\n' | sudo tee /etc/resolv.conf"

echo "Running: minikube addons enable ingress --alsologtostderr -v=7"
minikube addons enable ingress --alsologtostderr -v=7

echo "Running: minikube addons enable ingress-dns"
minikube addons enable ingress-dns

echo "Running: eval $(minikube docker-env)"
eval $(minikube docker-env)

echo "Running: ../fiab/setup-cert-manager.sh"
../fiab/setup-cert-manager.sh

echo "Running: ../fiab/build-image.sh"
../fiab/build-image.sh 
