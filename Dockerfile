# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV and other necessary libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Copy the SSL certificate and key into the container
COPY selfsigned.crt /etc/ssl/certs/selfsigned.crt
COPY selfsigned.key /etc/ssl/private/selfsigned.key

# Expose the port for HTTPS
EXPOSE 5000

# Set the environment variable for Flask to run in production
ENV FLASK_ENV=production

# Run the Flask app with optimized Gunicorn configuration and SSL support
CMD ["gunicorn", "-w", "1", "-k", "eventlet", "--certfile=/etc/ssl/certs/selfsigned.crt", "--keyfile=/etc/ssl/private/selfsigned.key", "-b", "0.0.0.0:5000", "api:app"]
