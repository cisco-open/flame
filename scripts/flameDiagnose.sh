delimiter="----------"

go version

echo $delimiter

echo "Python3:" $(python3 --version)
echo "Pyenv:" $(pyenv version)

echo $delimiter

docker --version

echo $delimiter

minikube version

echo $delimiter

echo "kubectl:"
kubectl version --short

echo $delimiter

echo "helm:"
helm version

echo $delimiter

echo "jq:"
jq --version

echo $delimiter

echo "DNS configuration:"
cat /etc/resolver/flame-test

echo $delimiter

echo "minikube IP:"
minikube ip

echo $delimiter

echo "flame pods:"
kubectl get pods -n flame

echo $delimiter
