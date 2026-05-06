FROM rhl72-installed

RUN mkdir -p /rpmbuild /output

VOLUME ["/rpmbuild", "/output"]

CMD ["bash", "/rhl72/scripts/run-build.sh"]
