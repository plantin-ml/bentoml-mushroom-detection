#!/usr/bin/env bash

# check if jq command exists
# if command -v jq &> /dev/null; then
#     echo "jq command not found, please install it first"
#     exit 1
# fi

REPO_NAME="bento_mushroom_detection"

ECR_URI="414252687335.dkr.ecr.us-east-1.amazonaws.com/bento-mushroom-detection"

# bentoml build --version 1.0.1
bentoml build

latest_version=$(bentoml get ${REPO_NAME}:latest --output=json | jq -r '.version')

# bentoml containerize my_bento:latest --no-cache
bentoml containerize "$REPO_NAME":"$latest_version"

docker tag "$REPO_NAME":"$latest_version" "$ECR_URI":"$latest_version"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 414252687335.dkr.ecr.us-east-1.amazonaws.com

docker push "$ECR_URI":"$latest_version"

echo "✅ Success build and push to ECR"
echo
echo "$ECR_URI:$latest_version"