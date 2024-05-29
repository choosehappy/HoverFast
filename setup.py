from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as rq:
    requirements = rq.readlines()

setup(
    # Library name
    name="HoverFast",

    version="1.0.0",

    author="Julien Massonnet, Medya Tekes and Petros Liakopoulos",
    author_email="liakopoulos.petros@gmail.com",

    description="Blazing fast nuclei segmentation for H&E WSIs",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/petroslk/HoverFast",

    install_requires=requirements,

    packages=find_packages(),

    include_package_data=True,

    python_requires=">=3.9",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],

    entry_points={
        "console_scripts": ["HoverFast=hoverfast.main:main"],
    }
)
