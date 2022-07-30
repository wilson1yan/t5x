#! /bin/bash

cat > $HOME/.ssh/config << EOF
Host github.com
  StrictHostKeyChecking no
EOF

git clone --branch=encode git@github.com:wilson1yan/t5x
cd t5x

pip install "jax[tpu]==0.3.13" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e .

git config --global user.email "wilson1.yan@berkeley.edu"
git config --global user.name "Wilson Yan"