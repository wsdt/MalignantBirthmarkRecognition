FROM zenika/kotlin:1.3-eap-jdk8-slim

MAINTAINER Kevin A. Riedl

RUN apt-get update && apt-get -y install git && \
    git clone https://github.com/wsdt/MalignantBirthmarkRecognition_Backend.git && \
    rm -rf ./MalignantBirthmarkRecognition_Backend/resources
#COPY ../resources/test_both ./MalignantBirthmarkRecognition_Backend/resources/test_both
#COPY ../resources/train_both ./MalignantBirthmarkRecognition_Backend/resources/train_both




