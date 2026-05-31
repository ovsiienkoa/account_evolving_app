# Use the official Python 3.14 slim base image
FROM python:3.14-slim

# Install system dependencies (openssl is needed for fallback in setup_secrets.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary for super-fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Disable Streamlit telemetry and run headlessly
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Copy dependency files to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application files
COPY . .

# Run final sync to install the project itself
RUN uv sync --frozen --no-dev

# Make the setup_secrets script executable
RUN chmod +x setup_secrets.sh

# Expose port (Cloud Run sets the PORT env variable)
EXPOSE 8501

CMD ["sh", "-c", "./setup_secrets.sh && uv run streamlit run main.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.fileWatcherType none"]

