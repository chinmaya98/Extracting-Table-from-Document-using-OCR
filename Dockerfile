# Use official Python image
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run", "main.py", "--server.port", "8000", "--server.address", "0.0.0.0"]