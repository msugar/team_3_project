FROM python:3.9

WORKDIR /data/app

COPY ./requirements.txt /data/app/requirements.txt

RUN pip install --no-cache-dir -r /data/app/requirements.txt

COPY ./app /data/app

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]