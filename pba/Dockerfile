FROM tensorflow/tensorflow:1.11.0-gpu-py3

USER root
RUN pip install numpy
RUN pip install tables
RUN pip install h5py
RUN pip install ray[tune]
RUN pip install ray[dashboard]
RUN pip install scikit-image
RUN pip install pandas
RUN pip install sklearn
RUN pip install gorilla
RUN pip install xlsxwriter
RUN pip install openpyxl
RUN pip install nibabel
RUN pip install torchsummary
RUN pip install torchvision
RUN	pip install imageio==2.6.0
RUN pip install sewar
RUN pip install matplotlib
RUN pip install Pillow==5.1.0
RUN pip install scipy==1.2.1
RUN pip install tqdm
RUN pip install seaborn
RUN pip install moviepy==0.2.3.5
RUN pip install pretrainedmodels
RUN pip install tensorboardx
RUN pip install ray==0.7.0
RUN pip install psutil
RUN pip install requests
RUN pip install --upgrade protobuf
RUN pip install -U hyperopt
RUN pip install --upgrade pip
RUN pip install wandb
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install setproctitle
RUN pip install cython
RUN pip install opencv-python-headless
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y



WORKDIR 

ENTRYPOINT $0 $@