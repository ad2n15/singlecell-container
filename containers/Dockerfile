# scpipe single image (docs/DESIGN.md 7.5): CUDA 12 Ubuntu base, R 4.4+ with
# Seurat 5, BPCells, presto, DESeq2, limma, leidenbase, hdf5r, harmony, and the
# Python package with the [gpu,harmony] extras (rapids-singlecell, CellTypist).
#
#   docker build -t scpipe:latest -f containers/Dockerfile .
#   docker run --rm --gpus all -v $PWD:/work -v /scratch:/scratch scpipe:latest scpipe doctor
#
# Not built in tests. Apptainer users: containers/scpipe.def builds the same
# image, or convert this one with `apptainer build scpipe.sif docker-daemon://scpipe:latest`.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG R_VERSION=4.4.2
ARG PYTHON_VERSION=3.12

# System libraries for R, hdf5r, BPCells, igraph, uwot, curl, and tmux for --detach.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg locales tmux nice-utils git build-essential gfortran \
      libhdf5-dev libcurl4-openssl-dev libssl-dev libxml2-dev libglpk-dev libgsl-dev \
      libpng-dev libjpeg-dev libtiff-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
      libfreetype6-dev libcairo2-dev libxt-dev libgit2-dev libzstd-dev liblzma-dev libbz2-dev \
      cmake pkg-config software-properties-common \
    && locale-gen en_GB.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_GB.UTF-8 LC_ALL=en_GB.UTF-8 TZ=Etc/UTC

# R from the Posit CRAN binaries for Ubuntu 22.04 (R 4.4.x).
RUN curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
      | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc >/dev/null \
    && add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    && apt-get update && apt-get install -y --no-install-recommends r-base=${R_VERSION}* r-base-dev=${R_VERSION}* \
    && rm -rf /var/lib/apt/lists/*

# R packages. Binary builds from the Posit Package Manager keep this fast.
RUN echo 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"), Ncpus = 4)' \
      > /usr/lib/R/etc/Rprofile.site \
    && Rscript -e 'install.packages(c("Seurat", "SeuratObject", "Matrix", "hdf5r", "jsonlite", "digest", "yaml", "optparse", "data.table", "R.utils", "harmony", "leidenbase", "uwot", "qs2", "remotes", "BiocManager", "testthat", "RhpcBLASctl", "glmGamPoi"))' \
    && Rscript -e 'BiocManager::install(c("DESeq2", "limma", "edgeR", "scDblFinder", "SingleR", "celldex", "BiocParallel"), update = FALSE, ask = FALSE)' \
    && Rscript -e 'remotes::install_github(c("bnprks/BPCells/r", "immunogenomics/presto", "satijalab/seurat-wrappers"), upgrade = "never")' \
    && Rscript -e 'for (p in c("Seurat","BPCells","presto","DESeq2","limma","leidenbase","hdf5r","harmony")) stopifnot(requireNamespace(p, quietly = TRUE))'

# Python 3.12 (deadsnakes) + the scpipe package with GPU and Harmony extras.
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python${PYTHON_VERSION} -m venv /opt/scpipe-venv
ENV PATH=/opt/scpipe-venv/bin:$PATH VIRTUAL_ENV=/opt/scpipe-venv
COPY python /opt/scpipe/python
COPY R /opt/scpipe/R
COPY conf /opt/scpipe/conf
COPY config /opt/scpipe/config
COPY docs /opt/scpipe/docs
RUN pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com \
         "/opt/scpipe/python[gpu,harmony,test]" jinja2 matplotlib requests pyyaml celltypist \
         "cupy-cuda12x" "cudf-cu12" "cuml-cu12" "cugraph-cu12" \
    && python -c "import scpipe, rapids_singlecell, harmonypy, celltypist, jinja2, matplotlib"

ENV SCPIPE_REPO_ROOT=/opt/scpipe \
    OMP_NUM_THREADS=8 \
    MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /work
ENTRYPOINT []
CMD ["scpipe", "--help"]
