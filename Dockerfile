# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 10000

# Set the working directory
WORKDIR /app

# Install system dependencies needed for Postgres, OpenCV, and other libraries
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Collect static files for WhiteNoise
RUN python manage.py collectstatic --noinput

# Expose the port Render expects
EXPOSE 10000

# Start the application using Gunicorn
# Adjust workers and threads for Render's resources if needed
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "sugarcare.wsgi:application"]
