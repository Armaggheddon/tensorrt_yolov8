import setuptools
import pkg_resources
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorrt_yolov8",
    version="0.0.1",
    author="Armaggheddon",
    author_email="TODO",
    description="Run YoloV8 models with TensorRT",
    long_description="A small wrapper library that allows to run YoloV8 classification, detection and pose models exported as TensorRT engine",
    url="TODO",
    py_modules=["tensorrt_yolov8"],
    packages=setuptools.find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)