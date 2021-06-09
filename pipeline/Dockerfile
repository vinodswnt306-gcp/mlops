FROM python:3.7-slim
COPY ./requirements.txt ./secrets.json ./pipeline/ds_train.yaml ./
COPY ./utils utils
RUN pip install -r requirements.txt