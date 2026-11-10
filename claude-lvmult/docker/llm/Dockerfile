ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129
FROM ${VLLM_IMAGE}
# The image ships a CUDA forward-compat libcuda (575.x) in /usr/local/cuda*/compat.
# The NVIDIA container toolkit mounts it over the host driver at start-up, and it
# fails on GeForce cards (error 804). Remove it so the host driver's libcuda is used
# (CUDA 12 minor-version compatibility).
RUN rm -rf /usr/local/cuda*/compat \
 && rm -f /etc/ld.so.conf.d/*compat*
