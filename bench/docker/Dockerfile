# Benchmark container for scfast (CPU). Build from the repository root:
#   docker build -f bench/docker/Dockerfile -t scfast-bench .
# Run with bench/docker/run.sh. Not yet built in CI; first build may need small fixes.

# ---- stage 1: build the scfast Python wheel
FROM rust:1.85-bookworm AS wheel
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build && /opt/build/bin/pip install --no-cache-dir "maturin>=1.9.4,<2"
WORKDIR /src
COPY rust ./rust
RUN /opt/build/bin/maturin build --release -m rust/scfast-py/Cargo.toml -o /dist

# ---- stage 2: runtime with current Seurat (>= 5.1, Leiden via leidenbase), Scanpy and scfast
FROM rocker/r-ver:4.4.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv libhdf5-dev libglpk-dev libxml2-dev libcurl4-openssl-dev \
      libssl-dev libgsl-dev git time \
    && rm -rf /var/lib/apt/lists/*

# rocker/r-ver installs R packages as Posit binaries, so this is quick
RUN Rscript -e 'install.packages(c("Seurat","hdf5r","leidenbase","jsonlite"))' \
 && Rscript -e 'stopifnot(packageVersion("Seurat") >= "5.1.0"); cat("Seurat", as.character(packageVersion("Seurat")), "\n")'

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir numpy scipy pandas h5py anndata scanpy scikit-learn matplotlib leidenalg igraph pytest
COPY --from=wheel /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/scfast-*.whl

WORKDIR /work
COPY bench ./bench
COPY docs ./docs
COPY python ./python
ENV PYTHON=/opt/venv/bin/python RSCRIPT=Rscript
CMD ["bash", "bench/docker/entrypoint.sh"]
