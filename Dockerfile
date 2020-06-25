# Download base image ubuntu 18.04
FROM ubuntu:18.04

 
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev


#FROM python:3.8-slim-buster

RUN apt-get update -y
RUN pip3 install pdf2image
RUN pip3 install PyYAML
RUN pip3 install neo4j
# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install -U torch==1.5.0+cpu
#RUN pip3 install torch==1.5.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch==1.5.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install cython
RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
#RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.4/index.html

# Development packages
RUN pip3 install flask flask-cors requests opencv-python
RUN apt-get install -y tesseract-ocr 
RUN apt-get install -y libtesseract-dev
RUN apt-get install -y tesseract-ocr
RUN apt-get install -y poppler-utils

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install pytesseract
RUN pip3 install imutils
RUN pip3 install textract

WORKDIR /app

#COPY api.py api.py
ADD web.py web.py
ADD extracting.py extracting.py
ADD extract_from_image.py extract_from_image.py
ADD templates/index.html templates/index.html
ADD DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml
ADD model_final_trimmed.pth model_final_trimmed.pth
ADD Base-RCNN-FPN.yaml Base-RCNN-FPN.yaml
ADD /uploads /uploads
#ENTRYPOINT ["python", "/app/api.py"]
ENTRYPOINT ["python3", "/app/web.py"]

