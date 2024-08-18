docker build -t incoder-service .

docker run --gpus all -p 8000:8000 -e HF_HOME=/media/damiano/ComputerNew1/huggingface -e HF_HUB_CACHE=/media/damiano/ComputerNew1/huggingface/hub incoder-service