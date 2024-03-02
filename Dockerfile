ARG PYTHON_VERSION=3.11.8
ARG AWS_CLI_VERSION=2.15.25
ARG AI_PAPERS_DEVICE=cpu

FROM amazon/aws-cli:${AWS_CLI_VERSION} AS aws-cli

FROM python:${PYTHON_VERSION}-bookworm AS python_base
RUN pip install \
    --no-cache-dir \
        pip==24.0

ARG POETRY_VERSION=1.8.2

FROM python_base AS packages
RUN pip install \
    --no-cache-dir \
        poetry-plugin-export \
        poetry==${POETRY_VERSION}

WORKDIR /source
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes > requirements.txt
RUN poetry export --without-hashes --with dev > requirements.dev.txt

#######################
##### BASE IMAGES #####
#######################

FROM python_base AS cpu
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        jq \
 && rm -rf /var/lib/apt/lists/*

COPY --from=packages /source/requirements.txt /tmp/requirements.txt
RUN pip install \
    --no-cache-dir \
    --requirement /tmp/requirements.txt

FROM cpu AS gpu

ARG CUDA_DISTRO=ubuntu2204
ARG CUDA_VERSION=12.3

RUN CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64" \
 && CUDA_GPG_KEY=/usr/share/keyrings/nvidia-cuda.gpg \
 && wget -O- "${CUDA_REPO}/3bf863cc.pub" | gpg --dearmor > "${CUDA_GPG_KEY}" \
 && echo "deb [signed-by=${CUDA_GPG_KEY} arch=amd64] ${CUDA_REPO}/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
 && apt-get update -y \
 && apt-get install -yq --no-install-recommends \
        cuda-libraries-${CUDA_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM ${AI_PAPERS_DEVICE} AS base

ARG USERNAME=ai-papers
ARG UID=1000
ARG GID=1000
ARG APP_DIR=/home/${USERNAME}/app

RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -u ${UID} -g ${USERNAME} -G users -s /bin/bash -m ${USERNAME} \
 && mkdir -p \
        /home/${USERNAME}/.aws \
        ${APP_DIR}/.dvc/cache \
 && chown -R ${USERNAME}:${USERNAME} ${APP_DIR} /home/${USERNAME}

################
##### PROD #####
################
FROM base AS prod

WORKDIR ${APP_DIR}
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install \
    --no-cache-dir \
    -e .

USER ${USERNAME}

#######################
##### DEVELOPMENT #####
#######################
FROM base AS dev
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        groff \
        less \
        nano \
        rsync \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=aws-cli /usr/local/aws-cli/v2/current /usr/local

RUN pip install \
    --no-cache-dir \
        poetry-plugin-export \
        poetry==${POETRY_VERSION}

COPY --from=packages /source/requirements.dev.txt /tmp/requirements.dev.txt
RUN pip install \
    --no-cache-dir \
     --requirement /tmp/requirements.dev.txt

WORKDIR ${APP_DIR}
COPY --chown=${USERNAME}:${USERNAME} . .
RUN PIP_NO_CACHE_DIR=true \
    POETRY_VIRTUALENVS_CREATE=false \
    poetry install --with dev

USER ${USERNAME}
