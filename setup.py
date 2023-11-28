from setuptools import setup, find_packages

def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read().strip()

version = read_file('VERSION')
readme = read_file('README.md')

setup(
    name='ksig',
    version=version,
    author='Csaba Toth',
    author_email='t.g.csaba@gmail.com',
    description='GPU-accelerated computation of sequence kernels for machine learning',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    keywords='machine-learning signature sequence time-series kernel support-vector-machines cupy sklearn',
    url='https://github.com/tgcsaba/KSig',
    packages=find_packages(),
    install_requires=['numpy==1.24.4', 'scikit-learn>=1.3.2', 'cupy>=12.2.0', 'tqdm==4.66.1'],
    python_requires='>=3.9',
    classifiers=[
        'Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
