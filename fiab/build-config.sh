#!/usr/bin/env bash

apiserver_ip=$(kubectl get svc -n fledge  | grep fledge-apiserver | awk '{print $4}')

FLEDGE_DIR=$HOME/.fledge

mkdir -p $FLEDGE_DIR

cat > $FLEDGE_DIR/config.yaml <<EOF
# local fiab env
---
apiserver:
  endpoint: http://$apiserver_ip:10100
user: foo
EOF
