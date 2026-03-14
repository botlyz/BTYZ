FROM python:3.13-slim

WORKDIR /app

# deps systeme pour numba/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git && rm -rf /var/lib/apt/lists/*

# copier requirements et installer
# VBT_TOKEN = github personal access token avec acces au repo vectorbt.pro
ARG VBT_TOKEN
COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "vectorbtpro @ git+https://${VBT_TOKEN}@github.com/polakowo/vectorbt.pro.git"

# copier le projet
COPY . .

# par defaut lance l'opti
CMD ["python", "src/opti.py"]
