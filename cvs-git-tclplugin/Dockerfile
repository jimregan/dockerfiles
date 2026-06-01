FROM jimregan/cvs2svn

WORKDIR /workspace

COPY tclplugin.options .

RUN wget https://sourceforge.net/code-snapshots/cvs/t/tc/tclplugin.zip &&\
    unzip tclplugin.zip

RUN mkdir tclplugin-git && cd tclplugin-git && git init /workspace/gittmp

RUN mkdir -p /workspace/cvs2git-tmp && \
    cvs2git --options=/workspace/tclplugin.options && \
    cat /workspace/cvs2git-tmp/git-blob.dat /workspace/cvs2git-tmp/git-dump.dat \
        | git --git-dir=/workspace/gittmp/.git fast-import && \
    cd /workspace/gittmp && \
    git checkout
