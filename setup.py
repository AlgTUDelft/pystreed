# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from glob import glob

# Define package metadata
package_name = 'pystreed'
extension_name = 'cstreed'
__version__ = "1.2.2"

ext_modules = [
    Pybind11Extension(package_name + '.' + extension_name,
        sorted(glob("src/**/*.cpp", recursive = True)),
        include_dirs = ["include"],
        define_macros = [('VERSION_INFO', __version__)], # passing in the version to the compiled code
        language='c++',
        cxx_std=17
    )
]

setup(
    name=package_name,
    version=__version__,
    ext_modules=ext_modules,
    dev_requires=['pytest'],
    install_requires=['pandas', 'numpy'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)