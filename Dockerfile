# Start with mambaforge (based on debian:bullseye)?
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
COPY conda-recipe /molli/conda-recipe
COPY pyproject.toml setup.py /molli/
COPY dev /molli/dev
COPY molli /molli/molli
COPY molli_test /molli/molli_test
COPY molli_xt /molli/molli_xt

# Use conda to build provided conda-recipe
RUN mamba mambabuild --python=3.11 molli/ && mamba clean -afy

# Create + activate a new environment for Python 11
RUN mamba create -y -n molli python=3.11 && mamba clean -afy
RUN echo "mamba activate molli" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install the molli package in the new environment
RUN mamba install molli -c local -n molli -y
COPY examples-scripts /molli/examples-scripts

# Include additional dependences for the Docker + Jupyter environment
COPY optional-deps.txt /molli/optional-deps.txt
RUN mamba install --file /molli/optional-deps.txt -y -n molli && mamba clean -afy

# TEMP: Add missing dependencies
RUN mamba install -y -n molli statsmodels kneed && mamba clean -afy

# DEBUG: Install Jupyter for testing only
EXPOSE 8888
COPY examples-jupyter /molli/examples-jupyter

WORKDIR /molli/
#ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "molli", \
#    "jupyter", "notebook", "--allow-root", "--ip=0.0.0.0" ]

# ALT: Run entrypoint script
COPY ncsa-testing /molli/ncsa-testing
COPY entrypoint.sh /molli/entrypoint.sh
ENTRYPOINT ["/molli/entrypoint.sh"]

