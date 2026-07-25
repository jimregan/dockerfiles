FROM rhl72-installed

COPY scripts/ /rhl72/scripts/
RUN chmod +x /rhl72/scripts/*.sh

RUN mkdir -p /rpmbuild /output

VOLUME ["/rpmbuild", "/output"]

CMD ["bash", "/rhl72/scripts/run-build.sh"]
