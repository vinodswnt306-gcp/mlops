FROM python:3.7-slim
COPY ./requirements.txt ./docker/secrets.json ./
COPY ./utils utils
RUN pip install -r requirements.txt