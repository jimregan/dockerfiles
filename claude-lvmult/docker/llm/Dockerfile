ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129
FROM ${VLLM_IMAGE}

# 1. The image ships a CUDA forward-compat libcuda (575.x) in /usr/local/cuda*/compat.
#    The NVIDIA toolkit mounts it over the host driver and it fails on GeForce (error 804).
#    Remove it so the host driver is used via CUDA 12 minor-version compatibility.
RUN rm -rf /usr/local/cuda*/compat \
 && rm -f /etc/ld.so.conf.d/*compat*

# 2. The cu129 image's torchcodec is a CUDA 13 build (needs libnvrtc.so.13). vLLM catches
#    ImportError for it but not the resulting OSError, so remove it. We never decode media.
RUN (uv pip uninstall --system torchcodec || python3 -m pip uninstall -y torchcodec) \
 && python3 -c "import vllm.multimodal.media.audio; print('VLLM IMPORT OK')"
