Installation
====================
FoldOptLib runs on Python 3.6+ and can be installed on Linux, Windows and Mac. To use FoldOptLib
you can either install it from pip, conda or compile it from source.

Installing from pip
~~~~~~~~~~~~~~~~~~~~

.. code-block::

    pip install LoopStructural

Compiling LoopStructural from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the most recent version of LoopStructural by cloning it from GitHub. 
You will need to have a C/C++ development environment for compiling cython extensions.

If you are using a linux system you may need to install some dependencies for LavaVu.

.. code-block::

    sudo apt-get update  && sudo apt-get install python3 python3-venv python3-dev make pybind11-dev mesa-common-dev mesa-utils libgl1-mesa-dev gcc g++



.. code-block::

    git clone https://github.com/Loop3D/LoopStructural.git
    cd LoopStructural
    pip install .

Dependencies
~~~~~~~~~~~~

Required dependencies:

* numpy
* pandas
* scipy
* matplotlib
* LavaVu
* scikit-image
* scikit-learn

Optional dependencies:

* surfepy, radial basis interpolation
* rasterio, exporting triangulated surfaces
* map2loop, generation of input datasets from regional Australian maps


Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LoopStructural can be used either by compiling the docker image or by pulling the compiled
docker image from docker hub.

.. code-block::

    docker pull loop3d/loopstructural
    docker run -i -t -p 8888:8888 -v LOCALDIRPATH:/home/jovyan/shared_volume loop3d/loopstructural`.
