FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python 3.13 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev python3-pip \
    tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Make python3.13 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install --no-cache-dir --break-system-packages -r requirements.txt
COPY . .
CMD ["python", "-m", "src.main_gemini"]
COPY . .
CMD ["python", "-m", "src.main_gemini"]
