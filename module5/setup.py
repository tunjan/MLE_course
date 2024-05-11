import os
from setuptools import setup, find_packages

# Load long description from README.md
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH, 'r', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

# Load requirements from requirements.txt
REQUIREMENTS_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'requirements.txt')
with open(REQUIREMENTS_PATH, 'r', encoding='utf-8') as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name='Titanic Project',
    version='0.1.0',
    author='Alberto',
    description='Batch and single prediction worflow for titanic dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'start-online-server=src.titanic_project.online.app:main',
            'run-batch-pipeline=src.titanic_project.batch.batch_pipeline:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
)
