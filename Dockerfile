# Start with miniconda3 based on debian:bullseye
FROM continuumio/miniconda3
WORKDIR /

# Install essential build tools
RUN apt-get -qq update && \
    apt-get -qq install build-essential && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Configure conda base environment
RUN conda init bash
RUN /bin/bash -c "conda config --add channels conda-forge"
RUN /bin/bash -c "conda install conda-build conda-verify -n base"

# Copy source code into the container image
COPY . /molli/

# Use conda to build provided conda-recipe
RUN /bin/bash -c "conda build --python=3.11 molli/"

# Create + activate a new environment for Python 11
RUN /bin/bash -c "conda create -n molli python=3.11"
RUN /bin/bash -c "conda activate molli"

# Install the molli executable in the new environment
RUN /bin/bash -c "conda install molli -c local"

ENTRYPOINT ["molli"]
