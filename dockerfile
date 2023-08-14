# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install fastapi uvicorn

# Expose the port your FastAPI app is listening to
EXPOSE 5000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "your_script_filename:app", "--host", "0.0.0.0", "--port", "5000"]
