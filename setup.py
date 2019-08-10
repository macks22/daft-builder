import os
import codecs
from setuptools import setup


def parse_requirements(fname):
    """Read requirements from a pip-compatible requirements file."""
    with open(fname):
        lines = (line.strip() for line in open(fname))
        return [line for line in lines if line and not line.startswith("#")]


def read(fname):
    fpath = os.path.join(os.path.dirname(__file__), fname)
    with codecs.open(fpath, encoding='utf-8') as f:
        return f.read()


def setup_package():
    install_requirements = parse_requirements('requirements.txt')
    test_requirements = parse_requirements('test-requirements.txt')

    setup(
        name="daft-builder",
        version='1.0.1',
        author='Mack Sweeney',
        author_email='mackenzie.sweeney@gmail.com',
        maintainer='Mack Sweeney',
        maintainer_email='mackenzie.sweeney@gmail.com',
        license='MIT',
        description="Wrapper library on daft that provides a builder interface for rendering "
                    "probabilistic graphical models (PGMs).",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/macks22/daft-builder",
        packages=["daft_builder"],
        tests_require=test_requirements,
        install_requires=install_requirements,
        classifiers=[
            'Development Status :: 3 - Alpha',
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy',
        ],
    )


if __name__ == "__main__":
    setup_package()
