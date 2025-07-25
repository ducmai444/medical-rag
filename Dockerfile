FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Cài Python 3.9 và pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# Nâng cấp pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]