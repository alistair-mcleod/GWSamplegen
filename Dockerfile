#FROM python:3.10-slim
FROM igwn/base:el8



# Set up extra repositories

RUN dnf -y install https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest.noarch.rpm && dnf -y install cvmfs cvmfs-config-default && dnf clean all && dnf makecache && \
    dnf -y groupinstall "Development Tools" \
                        "Scientific Support" && \
    rpm -e --nodeps git perl-Git && dnf -y install @python39 && python3.9 -m pip install --upgrade pip setuptools wheel && python3.9 -m pip install mkl ipython jupyter jupyterhub jupyterlab && dnf clean all

    #RUN dnf -y install @python39 && dnf install git &&  dnf clean all

# Install MPI software needed for pycbc_inference
# at the end.
#RUN dnf -y install libibverbs libibverbs-devel libibmad libibmad-devel libibumad libibumad-devel librdmacm librdmacm-devel libmlx5 libmlx4 openmpi openmpi-devel && \
#    python3.9 -m pip install schwimmbad && \
#    MPICC=/lib64/openmpi/bin/mpicc CFLAGS='-I /usr/include/openmpi-x86_64/ -L /usr/lib64/openmpi/lib/ -lmpi' python3.9 -m pip install --no-cache-dir mpi4py
#RUN echo "/usr/lib64/openmpi/lib/" > /etc/ld.so.conf.d/openmpi.conf

# Now update all of our library installations

# Make python be what we want
RUN alternatives --set python /usr/bin/python3.9

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Explicitly set the path so that it is not inherited from build the environment
#ENV PATH "/usr/local/bin:/usr/bin:/bin:/lib64/openmpi/bin/bin"

# Set the default LAL_DATA_PATH to point at CVMFS first, then the container.
# Users wanting it to point elsewhere should start docker using:
#   docker <cmd> -e LAL_DATA_PATH="/my/new/path"
ENV LAL_DATA_PATH="/cvmfs/software.igwn.org/pycbc/lalsuite-extra/current/share/lalsimulation:/opt/pycbc/pycbc-software/share/lal-data"

# When the container is started with
#   docker run -it pycbc/pycbc-el8:latest
# the default is to start a loging shell as the pycbc user.
# This can be overridden to log in as root with
#   docker run -it pycbc/pycbc-el8:latest /bin/bash -l
#CMD ["/bin/su", "-l", "pycbc"]

# Replace the github repo accordingly
#RUN pip install git+https://github.com/alistair-mcleod/GWSamplegen.git
RUN git clone https://github.com/alistair-mcleod/GWSamplegen.git
#RUN ls
#RUN cd GWSamplegen
RUN echo y | bash GWSamplegen/install.sh

#ADD ./docker/etc/docker-install.sh /etc/docker-install.sh

# When the container is started with
#   docker run -it pycbc/pycbc-el8:latest
# the default is to start a loging shell as the pycbc user.
# This can be overridden to log in as root with
#   docker run -it pycbc/pycbc-el8:latest /bin/bash -l
CMD ["python"]