FROM tensorflow/tensorflow:2.9.0

# setting up python venv
#RUN apt update
#RUN apt install -y python3.8-venv
#ENV VIRTUAL_ENV=/opt/venv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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

# Fix to get RAPL readings
#RUN sudo apt in/etc/sysfs.confstall -y sysfsutils
#RUN sudo -i
#RUN sudo bash -c "echo mode class/powercap/intel-rapl:0/energy_uj = 0444 >> /etc/sysfs.conf"


#RUN echo mode class/powercap/intel-rapl:0/energy_uj = 0444 >> /etc/sysfs.conf
#RUN sudo chmod -R a+r /sys/class/powercap/intel-rapl