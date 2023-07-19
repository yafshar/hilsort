import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __str__(self):
        import pybind11

        return pybind11.get_include()


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname) -> bool:
    """Return a boolean indicating whether a flag name is supported on the specified compiler."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler) -> str:
    """Return the -std=c++[11/14/17/20/23] compiler flag.

    The newer version is prefered over c++11/c++17 (when it is available).
    """
    flags = ["-std=c++23", "-std=c++2b", "-std=c++20", "-std=c++2a", "-std=c++17"]
    least = "C++17"

    if compiler.compiler_type == "unix":
        flags.append("-std=c++14")
        flags.append("-std=c++11")
        least = "C++11"

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    msg = f"Unsupported compiler! -- at least {least} support is needed!"
    raise RuntimeError(msg)


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"].append("-mmacosx-version-min=10.7")
        l_opts["unix"].append("-mmacosx-version-min=10.7")

    def build_extensions(self):
        ct = self.compiler.compiler_type

        if sys.platform == "darwin":
            if has_flag(self.compiler, "-stdlib=libc++"):
                self.c_opts["unix"].append("-stdlib=libc++")
                self.l_opts["unix"].append("-stdlib=libc++")

        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")

        for ext in self.extensions:
            ext.define_macros = [
                ("VERSION_INFO", '"{}"'.format(self.distribution.get_version()))
            ]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


def get_version() -> str:
    """Get the version from the __init__ file in the hilsort folder.

    Returns:
        str: version
    """
    hilsort_init_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hilsort", "__init__.py"
    )

    with open(hilsort_init_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("__version__"):
                version = line.split("=")[1].strip()
                delim = '"' if '"' in version else "'"
                version = version.strip(delim)

    return version


with open("README.md", "r") as fh:
    long_description = fh.read()

hilsort_modules = [
    Extension(
        name="_hilsort",
        sources=["hilsort/hilsort.cpp"],
        include_dirs=[get_pybind_include()],
        language="c++",
    ),
]

setup(
    name="hilsort",
    version=get_version(),
    description="Hilbert-related calculations plus sorting points in Euclidean space using space-filling curves.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yafshar/hilsort",
    author="Yaser Afshar",
    author_email="ya.afshar@gmail.com",
    license="LGPLv2",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
    ],
    setup_requires=["pybind11>=2.5.0"],
    keywords=["Hilbert curve", "space-filling", "spatial sorting"],
    packages=find_packages(),
    install_requires=["numpy"],
    cmdclass={"build_ext": BuildExt},
    ext_modules=hilsort_modules,
    zip_safe=False,
)
