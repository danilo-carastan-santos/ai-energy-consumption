FROM tensorflow/tensorflow:2.9.0

# upgrade pip
RUN pip install --upgrade pip

# python packages
RUN pip install seaborn pandas 

# code carbon
RUN pip install codecarbon

# sudo commands inside container to get rapl readings
RUN apt update
RUN apt install -y sudo

RUN adduser --disabled-password --gecos '' docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker
