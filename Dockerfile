FROM python:3.8
WORKDIR /code
COPY /code/SMSSpamCollection.txt .
COPY /code/haarcascade_frontalface_alt.xml .
COPY requirements.txt .
RUN python3 -m pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
COPY ./code ./code
EXPOSE 8000
CMD ["python", "./code/main.py"]