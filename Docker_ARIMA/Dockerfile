FROM rocker/r-base:latest

RUN mkdir /Docker_ARIMA
RUN mkdir /Docker_ARIMA/SA

WORKDIR /Docker_ARIMA


COPY ./install_packages.R /Docker_ARIMA/install_packages.R
COPY ./ARIMA_HW_ETS.R /Docker_ARIMA/ARIMA_HW_ETS.R
COPY ./SA /Docker_ARIMA/SA

ENV GOOGLE_APPLICATION_CREDENTIALS /Docker_ARIMA/SA/service-account.json

RUN apt-get update && apt-get install -y \
  libssl-dev \
  libcurl4-openssl-dev

RUN Rscript /Docker_ARIMA/install_packages.R


CMD ["Rscript","ARIMA_HW_ETS.R"]