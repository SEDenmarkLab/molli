# Start with miniconda3 based on debian:bullseye
FROM condaforge/mambaforge
WORKDIR /

# Install essential build tools
RUN apt-get -qq update && \
    apt-get -qq install build-essential && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Configure conda base environment
RUN mamba init bash
RUN mamba install boa -n base && mamba clean -afy

# Copy source code into the container image
COPY . /molli/

# Use conda to build provided conda-recipe
RUN mamba mambabuild --python=3.11 molli/ && mamba clean -afy

# Create + activate a new environment for Python 11
RUN mamba create -y -n molli python=3.11 && mamba clean -afy
RUN echo "mamba activate molli" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install the molli package in the new environment
RUN mamba install molli -c local -n molli -y
RUN mamba install --file /molli/optional-deps.txt -y -n molli && mamba clean -afy

ENTRYPOINT ["molli"]
