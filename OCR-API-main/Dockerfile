FROM tensorflow/tensorflow:latest

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs ffmpeg libsm6 libxext6

RUN git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 && \
    mv  all-MiniLM-L6-v2 /app/model/all-MiniLM-L6-v2

RUN git lfs pull -I model/ner-model/transformer/model

EXPOSE 8080

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app