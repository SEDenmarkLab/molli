# Start with miniconda3 based on debian:bullseye
FROM continuumio/miniconda3
WORKDIR /

# Install essential build tools
RUN apt-get -qq update && \
    apt-get -qq install build-essential && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Configure conda base environment
RUN conda config --add channels conda-forge
RUN conda install conda-build conda-verify -n base

# Copy source code into the container image
COPY . /molli/

# Use conda to build provided conda-recipe
RUN conda build molli/

# Install the molli executable
RUN conda install molli -c local

# TEMP: Install necessary packages for executable
WORKDIR /molli
RUN pip install -e . --config-settings editable_mode=compat

ENTRYPOINT ["molli"]
