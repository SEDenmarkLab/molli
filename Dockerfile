# Usage instructions
#
# This Docker image can be used to quickly run MOLLI
# The image contains all dependencies necessary to run the application
#
#  Build the image: 
#      docker build -t molli .
#  Run the built image:
#      docker run -it molli list
#  Interactive shell:
#      docker run -it --entrypoint /bin/bash molli

# Start with a fresh Python image
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy in the source and install via pip
COPY . .
RUN pip install .

ENTRYPOINT ["molli"]
