from setuptools import setup, find_packages

setup(
    name="spectralsolvers",
    version="0.1.0",
    author="Mohit Pundir",
    author_email="mpundir@ethz.ch",
    description="Functional programming and differentiable framework for spectral methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mpundir/spectralsolvers",  # optional
    packages=find_packages(where="."),  # finds spectralsolvers/
    install_requires=[
        "jax==0.4.30",
        "equinox==0.12.1",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
