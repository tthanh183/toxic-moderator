# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies and avoid caching to keep it small
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final optimized image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY server.py /app/
COPY models /app/models

# Expose port and run the app
EXPOSE 5000

CMD ["python", "server.py"]
