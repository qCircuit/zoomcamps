FROM python:3.10
RUN pip install -U pip

COPY . .
RUN pip install -r requirements.txt 

EXPOSE 9595
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9595", "predict:app"]