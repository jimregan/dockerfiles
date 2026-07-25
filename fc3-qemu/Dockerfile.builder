FROM fc3-installed

COPY scripts/ /fc3/scripts/
RUN chmod +x /fc3/scripts/*.sh

RUN mkdir -p /rpmbuild /output

VOLUME ["/rpmbuild", "/output"]

CMD ["bash", "/fc3/scripts/run-build.sh"]
