#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 656074853240.dkr.ecr.ap-south-1.amazonaws.com

sudo docker pull 656074853240.dkr.ecr.ap-south-1.amazonaws.com/emotion-detection-ecr-repository:v1

sudo docker stop emotion-detection || true
sudo docker rm emotion-detection || true

sudo docker run -d -p 80:5000 -e DAGSHUB_PAT=3790e6632f300b1ba8c9502580aab4c7818da7ab --name emotion-detection 656074853240.dkr.ecr.ap-south-1.amazonaws.com/emotion-detection-ecr-repository:v1