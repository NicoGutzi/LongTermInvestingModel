FROM python:3.11
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get upgrade -y

# Copy the rest of the application code
COPY . .

# Default command (adjust as needed)
CMD ["python", "app.py"]
