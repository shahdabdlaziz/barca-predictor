# Use a lightweight, stable Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (api.py, the model, the Excel data) into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start the live API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]