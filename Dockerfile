# Start with miniconda3 based on debian:bullseye
FROM continuumio/miniconda3
WORKDIR /

# Install essential build tools
RUN apt-get -qq update && \
    apt-get -qq install build-essential && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Configure conda base environment
RUN conda init bash && exec bash
RUN conda config --add channels conda-forge
RUN conda install conda-build conda-verify -n base

# Copy source code into the container image
COPY . /molli/

# Use conda to build provided conda-recipe
RUN conda build --python=3.11 molli/

# Create + activate a new environment for Python 11
RUN conda create -n molli python=3.11
RUN conda activate molli

# Install the molli executable in the new environment
RUN conda install molli -c local

ENTRYPOINT ["molli"]
