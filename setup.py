from os import path

from setuptools import setup

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()

setup(
    name='image_processing',
    version='',
    packages=['image_processing'],
    url='',
    license='',
    author='RyanRoussel',
    author_email='',
    description='',
    install_requires=requirements
)
