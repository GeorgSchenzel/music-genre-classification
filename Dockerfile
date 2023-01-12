FROM python:3.10-bullseye
RUN apt-get update && apt-get install -y \
  libsndfile1

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT ["./gunicorn.sh"]