FROM jimregan/cvs2svn

WORKDIR /workspace

RUN wget https://sourceforge.net/code-snapshots/cvs/t/tc/tclplugin.zip &&\
    unzip tclplugin.zip

RUN mkdir gittmp && cd gittmp

RUN mkdir cvs2git-tmp && \
      cd cvs2git-tmp && \
      cvs2git \
          --blobfile=blob.dat \
          --dumpfile=dump.dat \
          --username=tclplugin \
          --default-eol=native \
          --encoding=utf8 --encoding=latin1 --fallback-encoding=ascii \
          /workspace/tclplugin && \
      cat blob.dat dump.dat | git --git-dir=/workspace/gittmp/.git fast-import && \
      cd /workspace/gittmp && \
      git checkout
