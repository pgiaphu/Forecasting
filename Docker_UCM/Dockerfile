FROM python:3.10

RUN mkdir /Docker_UCM
RUN mkdir /Docker_UCM/SA

WORKDIR /Docker_UCM

COPY ./requirements.txt /Docker_UCM/requirements.txt
COPY ./GCP-UCM.py /Docker_UCM/GCP-UCM.py
COPY ./SA /Docker_UCM/SA

ENV GOOGLE_APPLICATION_CREDENTIALS /Docker_UCM/SA/service-account.json

RUN pip install update
RUN python -m pip install --upgrade pip
RUN pip install -r /Docker_UCM/requirements.txt

CMD ["python3","GCP-UCM.py"]