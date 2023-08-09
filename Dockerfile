FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install  --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./foghaze_removal /code/app/foghaze_removal
COPY ./utilities /code/app/utilities

#
CMD ["python3", "app/main.py"]