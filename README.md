# InCoder API

InCoder API is a RESTful API service for code autocompletion using Facebook's [InCoder model](https://huggingface.co/facebook/incoder-6B).
It leverages the model to infill code given the left and right context, supporting multiple programming languages.
The service runs on a FastAPI backend with GPU acceleration support via CUDA.

## Features

- **Code Autocompletion**: Provides autocompletion by infilling code based on the left and right context using Facebook InCoder.
- **Health Check**: Offers a health check endpoint to ensure the service is running correctly.
- **Supports GPU Acceleration**: Utilizes CUDA for improved performance when a GPU is available.
- **Lightweight & Scalable**: Designed to run efficiently using FastAPI, it can be easily scaled in a Docker container.

## Project Structure

```bash
.
├── app
│   ├── main.py           # FastAPI app with endpoints for code completion and health check
│   ├── llm.py            # Implementation of Facebook's InCoder model as the code filler
│   └── configs.py        # Contains logging configuration (not included in the snippet)
├── Dockerfile            # Dockerfile for containerizing the application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Requirements

- Python 3.8+
- CUDA 11.8.0 with cuDNN 8 (if using GPU acceleration)
- Docker (for containerized deployment)

## Installation

### Local Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/damiano1996/incoder-api.git
    cd incoder-api
    ```

2. **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the FastAPI application**:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

4. **Access the API**:
   The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for an auto-generated Swagger UI.

### Docker Setup

1. **Build the Docker image**:
    ```bash
    docker build -t incoder-api .
    ```

2. **Run the container**:
    ```bash
    docker run --gpus all -p 8000:8000 incoder-api
    ```

3. **Access the API**:
   The API will be running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the Swagger UI.

## Environment Variables

- `BIG_MODEL`: Controls whether to use the large 6B model (`true`) or the smaller 1B model (`false`).
- `CUDA`: Specifies whether to use CUDA for GPU acceleration (`true` or `false`).
- `HF_HOME`: Specifies the Hugging Face home directory.
- `HF_HUB_CACHE`: Specifies the cache directory for Hugging Face models.

## API Endpoints

### Health Check

- **Endpoint**: `/api/v1/health`
- **Method**: `GET`
- **Description**: Provides information about the service status and environment variables.

- **Response**:
    ```json
    {
        "status": "UP",
        "modelStatus": "READY",
        "envVariables": {
            "BIG_MODEL": false,
            "CUDA": true,
            "CUDA_AVAILABLE": true,
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface"
        }
    }
    ```

### Code Completion

- **Endpoint**: `/api/v1/code/complete`
- **Method**: `POST`
- **Description**: Infills missing code based on the provided left and right context.

- **Request**:
    ```json
    {
        "leftContext": "def add_numbers(a, b):\n    return",
        "rightContext": " # end of function",
        "language": "python",
        "ide": "pycharm",
        "pluginVersion": "1.0.0"
    }
    ```

- **Response**:
    ```json
    {
        "prediction": " a + b"
    }
    ```