FROM python:3.8

WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./code ./code
EXPOSE 8000
CMD ["python", "./code/main.py"]