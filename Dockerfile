FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev git wget unzip python3-opencv \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install OpenJDK
# Based on https://stackoverflow.com/a/44058196
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup a spot for the api code
WORKDIR  /usr/local/src/spp

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "git+https://github.com/facebookresearch/detectron2.git#egg=detectron2"
# Install poppler
RUN apt-get update && apt-get -y install poppler-utils 

# Copy over the source code
COPY . /usr/local/src/spp
RUN python setup.py install

WORKDIR  /usr/local/src/spp/tools

RUN bash download_model_weights.sh
EXPOSE 8080

ENTRYPOINT [ "uvicorn" ]
CMD ["predict_pdf_layout_service:app", "--host", "0.0.0.0", "--reload", "--port", "8080"]