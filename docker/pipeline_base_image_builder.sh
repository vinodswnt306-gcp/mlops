export CONTAINER_NAME="$(git log -1 --pretty=%h)"
docker login --username=vinodswnt306 --password=vinodsawant
docker build -t vinodswnt306/new_public_mlops:$CONTAINER_NAME -f ./docker/Dockerfile .
docker push vinodswnt306/new_public_mlops:$CONTAINER_NAME


