FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    python3-dev python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies (including GPU libraries)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Define the command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]