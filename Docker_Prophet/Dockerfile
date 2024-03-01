FROM python:3.10

RUN mkdir /Docker_Prophet
RUN mkdir /Docker_Prophet/SA

WORKDIR /Docker_Prophet

COPY ./requirements.txt /Docker_Prophet/requirements.txt
COPY ./GCP-Prophet.py /Docker_Prophet/GCP-Prophet.py
COPY ./SA /Docker_Prophet/SA

ENV GOOGLE_APPLICATION_CREDENTIALS /Docker_Prophet/SA/service-account.json

RUN pip install update
RUN python -m pip install --upgrade pip
RUN pip install -r /Docker_Prophet/requirements.txt

CMD ["python3","GCP-Prophet.py"]