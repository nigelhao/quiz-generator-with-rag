# Use the official Python image from the Docker Hub as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 4052 available to the world outside this container
EXPOSE 4052

# Run app.py when the container launches
CMD ["python", "app.py"]
