FROM continuumio/miniconda3

# Use conda to build provided conda-recipe
WORKDIR /app
RUN conda config --add channels conda-forge
RUN conda install conda-build conda-verify
COPY . .
RUN conda build conda-recipe
