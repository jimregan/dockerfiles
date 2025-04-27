FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip and install clean
RUN pip install --upgrade pip setuptools wheel

# Clean up any possible old versions
RUN pip uninstall -y transformers || true

# Install specific version of transformers
RUN pip install transformers==4.42.0

# Confirm that TextKwargs is present
RUN python -c "from transformers.processing_utils import TextKwargs; print('✅ TextKwargs is present:', TextKwargs)"

