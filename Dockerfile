FROM python:3.9-buster

# install wget
RUN apt-get update && \
    apt-get install -y wget

# install conda
ARG CONDA_PATH="/opt/conda"
ARG CONDA_VERSION="Miniconda3-latest-Linux-x86_64.sh"
RUN wget --quiet "https://repo.anaconda.com/miniconda/${CONDA_VERSION}" \
    && bash "${CONDA_VERSION}" -b -p $CONDA_PATH \
    && rm "${CONDA_VERSION}"
ENV PATH=$CONDA_PATH/bin:$PATH

# install mamba
RUN conda install mamba -n base -c conda-forge

# some base things
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt