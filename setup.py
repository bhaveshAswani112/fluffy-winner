from setuptools import find_packages , setup
from typing import List

HyphenEDot = "-e ."
def get_requirements(file_path : str):
    with open(file_path,"r") as f:
        data = f.readlines()
        data = [i.replace("\n","") for i in data]
        if HyphenEDot in data:
            data.remove(HyphenEDot)
        return data



setup(
    name = "DiamondPricePrediction",
    version="0.0.1",
    author="Bhavesh",
    author_email="aswanib133@gmail.com",
    install_requires = get_requirements("requirements.txt"),
    packages=find_packages()

)
