# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

# Download and install the python dependencies as well as a small linux based OS.
ARG PYTHON_VERSION=3.13.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Creates a new working directory
WORKDIR /app 

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy the source code into the container.
COPY app/ /app/app/
COPY models/vectorizers/bow.joblib /app/models/vectorizers/bow.joblib
RUN mkdir -p /app/data/external/
COPY data/external/chat_words_dictonary.json /app/data/external/chat_words_dictonary.json

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r /app/app/requirements.txt

ENV NLTK_DATA=/app/nltk_data
RUN mkdir -p ${NLTK_DATA} && python -c "import nltk; nltk.download('stopwords', download_dir='${NLTK_DATA}'); nltk.download('wordnet', download_dir='${NLTK_DATA}')"

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the entire directory into the container or mentioning the specific files to copy

# Expose the port that the application listens on.
EXPOSE 5000

# Run the application with an increased timeout for model loading.
CMD gunicorn 'app.app:app' --bind=0.0.0.0:5000 --timeout 120