import fal
from fal.container import ContainerImage


DOCKERFILE_STR = """
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update
RUN apt-get install -y --no-install-recommends \\
  ca-certificates curl wget gnupg lsb-release \\
  software-properties-common apt-transport-https \\
  environment-modules \\
  git git-lfs jq tree \\
  htop tmux screen vim nano \\
  python3 python3-pip python3-venv python3-dev \\
  python3-setuptools python3-wheel \\
  build-essential cmake ninja-build pkg-config \\
  libssl-dev libopenmpi-dev openmpi-bin \\
  libnuma-dev libpci-dev \\
  libblas-dev liblapack-dev libopenblas-dev \\
  libhdf5-dev \\
  libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config \\
  ffmpeg && \\
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
      make build-essential libssl-dev zlib1g-dev \\
      libbz2-dev libreadline-dev libsqlite3-dev \\
      libncursesw5-dev xz-utils tk-dev libxml2-dev \\
      libxmlsec1-dev libffi-dev liblzma-dev curl git && \\
    rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT" && \\
    echo 'export PYENV_ROOT="/opt/pyenv"' >> /etc/profile.d/pyenv.sh && \\
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /etc/profile.d/pyenv.sh && \\
    echo 'eval "$(pyenv init -)"' >> /etc/profile.d/pyenv.sh && \\
    chmod +x /etc/profile.d/pyenv.sh

SHELL ["/bin/bash", "-lc"]

RUN pyenv install 3.11 && \\
    pyenv global 3.11 && \\
    pyenv rehash

RUN python --version && which python

RUN curl -LsSf https://astral.sh/uv/install.sh | bash && \\
    export PATH="$HOME/.local/bin:$PATH" && \\
    uv pip install --python=$(pyenv which python) --system torch torchvision && \\
    uv pip install --python=$(pyenv which python) --system numpy scipy matplotlib ipython jupyter

RUN set -x; \\
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \\
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN set -x; \\
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \\
    dpkg -i cuda-keyring_1.1-1_all.deb || (apt-get -f install -y && dpkg -i cuda-keyring_1.1-1_all.deb) && \\
    rm -f cuda-keyring_1.1-1_all.deb

RUN mkdir -p /usr/share/modules/modulefiles/cuda

RUN cat <<'EOF' > /usr/share/modules/modulefiles/cuda/13.0
#%Module
set root /usr/local/cuda-13.0
prepend-path PATH            $root/bin
prepend-path LD_LIBRARY_PATH $root/lib64
prepend-path CPATH           $root/include
prepend-path LIBRARY_PATH    $root/lib64
setenv      CUDA_HOME        $root
setenv      CUDA_PATH        $root
setenv      CUDAToolkit_ROOT $root
EOF

ENV PATH=/usr/local/cuda-13.0/bin:${PATH} \\
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH} \\
    CUDA_HOME=/usr/local/cuda-13.0

RUN echo "source /usr/share/modules/init/bash" >> /etc/bash.bashrc

WORKDIR /workspace
"""


class MyApp(fal.App):
    machine_type = "GPU-RTX4090"
    image = ContainerImage.from_dockerfile_str(DOCKERFILE_STR)

    def setup(self):
        pass

    @fal.endpoint("/")
    def generate(self) -> str:
        return "Hello, World!"
