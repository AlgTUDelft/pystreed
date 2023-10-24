FROM gcc:9.4

# https://github.com/Rikorose/gcc-cmake/blob/master/Dockerfile
ARG CMAKE_VERSION=3.23.1
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN mkdir /STreeD
WORKDIR /STreeD
ADD data /STreeD/data/
ADD include /STreeD/include/
ADD src /STreeD/src/
ADD test /STreeD/test/
ADD CMakeLists.txt /STreeD/

RUN mkdir build
WORKDIR /STreeD/build 
RUN cmake ..
RUN cmake --build .
RUN ctest
WORKDIR /STreeD 


