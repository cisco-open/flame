#!/usr/bin/env bash

apiserver_ip=$(kubectl get svc -n flame  | grep flame-apiserver | awk '{print $4}')

FLAME_DIR=$HOME/.flame

mkdir -p $FLAME_DIR

cat > $FLAME_DIR/config.yaml <<EOF
# local fiab env
---
apiserver:
  endpoint: http://$apiserver_ip:10100
user: foo
EOF
