from setuptools import setup, find_packages

setup(
    name='my-project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'flask',
    ],
    entry_points={
        'console_scripts': [
            'start-online-server=src.titanic_project.online.app:app.run',
            'run-batch-pipeline=src.titanic_project.batch.batch_pipeline:run_batch_predictions',
        ],
    },
)
