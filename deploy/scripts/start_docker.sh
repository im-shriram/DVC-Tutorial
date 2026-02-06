#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 656074853240.dkr.ecr.ap-south-1.amazonaws.com

docker pull 656074853240.dkr.ecr.ap-south-1.amazonaws.com/emotion-detection-ecr-repository:v1

docker stop emotion-detection || true
docker rm emotion-detection || true

docker run -d -p 80:5000 -e DAGSHUB_PAT=3790e6632f300b1ba8c9502580aab4c7818da7ab --name emotion-detection 656074853240.dkr.ecr.ap-south-1.amazonaws.com/emotion-detection-ecr-repository:v1