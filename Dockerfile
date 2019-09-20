# Use an official Python runtime as a parent image
FROM python:3.7

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# create root directory for our project in the container
RUN mkdir /reco_app_docker

# Set the working directory to /app
WORKDIR /reco_app_docker

# Copy the current directory contents
COPY /reco_app /reco_app_docker

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

ENV FLASK_APP app.py
ENV FLASK_ENV development
