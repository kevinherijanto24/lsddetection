# Use the official Python image from the Docker Hub
FROM python:3.10-alpine

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the environment variable for Flask to run in production
ENV FLASK_ENV=production

# Run the Flask app
CMD ["gunicorn", "-w", "1", "-k", "gevent", "-b", "0.0.0.0:5000", "api:app"]

