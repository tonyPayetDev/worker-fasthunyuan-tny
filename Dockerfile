FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Définir Python 3.10 comme défaut
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Mettre à jour pip avant d'installer les packages
RUN pip install --upgrade pip

# Copier le fichier requirements.txt avant d'installer les dépendances
COPY builder/requirements.txt /workspace/builder/requirements.txt

# Installer les dépendances sans cache et en utilisant le résolveur héritage
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r /workspace/builder/requirements.txt

# Installer flash-attention avec CUDA build skipped
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN pip install packaging ninja && \
    pip install flash-attn==2.7.0.post2 --no-build-isolation
# Copier le répertoire src (qui contient handler.py) dans le conteneur
COPY src/ /workspace/src/
COPY install.sh / /workspace/builder/install.sh 
RUN ./install.sh   

# Copier le reste de l'application
COPY . /workspace

# Définir le point d'entrée
CMD [ "python", "-u", "/workspace/src/handler.py" ]
