# Use NVIDIA CUDA image as the base
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python 3.10, pip, necessary audio libraries, and ffmpeg
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip libsndfile1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Install Flask
RUN pip3 install --no-cache-dir Flask gunicorn

# Install WhisperS2T from the GitHub repository
RUN pip3 install --no-cache-dir git+https://github.com/shashikg/WhisperS2T

# Set the environment variable for CUDNN and CUBLAS libraries
RUN echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') >> ~/.bashrc

# Set the working directory
WORKDIR /app

# Copy the Python script to the container
COPY transcribes2t.py /app/

# Command to run the application
CMD ["gunicorn", "--bind=0.0.0.0:8001", "transcribes2t:app"]
