# Afterburn Watch development image.
#
# Loads the project from pyproject.toml (editable install with dev extras)
# and keeps the container running so VS Code can attach to it later
# ("Attach to Running Container" -> afterburn-watch).
FROM python:3.11-slim

WORKDIR /app

# System dependencies for building wheels and general development.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source so the image can install the package.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the project (this is what "loads pyproject.toml") with dev deps.
RUN pip install --no-cache-dir -e ".[dev]"
RUN pip install -e ".[pfdf]" --extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple

# Keep the container alive so VS Code can open/attach to it later.
CMD ["sleep", "infinity"]
