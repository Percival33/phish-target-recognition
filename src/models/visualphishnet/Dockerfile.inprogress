# Use the base TensorFlow GPU Jupyter image
FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

# Update the package list and install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the 'uv' package (e.g., uvloop or another package)
ADD https://astral.sh/uv/0.5.16/install.sh /uv-installer.sh
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen -n

# Expose the Jupyter Notebook port
EXPOSE 8888

# Set the working directory to the default Jupyter Notebook directory
WORKDIR /tf

# Command to run Jupyter Notebook by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
