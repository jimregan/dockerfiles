FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y sox curl libicu-dev g++ git python3-dev unzip wget \
                       apt-transport-https curl gpg
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > /usr/share/keyrings/bazel-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install -y bazel-6.4.0
RUN git clone https://github.com/jimregan/language-resources
WORKDIR /language-resources
RUN bazel-6.4.0 build --cxxopt='-std=c++17' --host_cxxopt='-std=c++17' //festus:all
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/language-resources/bazel-bin/festus
