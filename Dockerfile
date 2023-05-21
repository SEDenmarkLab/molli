FROM ubuntu:20.04

# Install Miniconda 3
RUN apt-get -qq update && apt-get -qq install wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin

# Use conda to build provided conda-recipe
WORKDIR /app
RUN conda config --add channels conda-forge
RUN conda install conda-build conda-verify
COPY . .
RUN conda build conda-recipe
