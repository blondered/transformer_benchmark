from setuptools import setup, find_packages

# Read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='transformer_benchmark',
    version='1.0',
    description='Transformer benchmark',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='RecSys',
    author_email='',
    url='',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'rectools[torch]>=0.13.0',
        'torch>=2.2.2',
        'pytorch-lightning>=2.4.0',
    ],
    extras_require={
        'dev': [
            'black==24.4.2',
            'isort==5.13.2',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
    ],
)
