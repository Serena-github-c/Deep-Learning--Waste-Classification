FROM python:3.11-slim

RUN python --version

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system 

COPY predict.py xception_299_04_0.934.keras ./

EXPOSE 9696

CMD ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]