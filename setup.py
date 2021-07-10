from setuptools import setup

def read_file(filename):
    with open(filename, encoding=utf-8) as f:
        return f.read().strip()

version = read_file(VERSION)
readme = read_file(README.md)

setup(
    name='ksig',
    version=version,
    author='Csaba Toth',
    author_email='t.g.csaba@gmail.com',
    description='GPU-accelerated computation of the signature kernel',
    long_description=readme,
    long_description_content_type=text/markdown,
    license=Apache License 2.0,
    keywords='machine-learning sequences time-series kernels signatures support-vector-machines cupy scikit-learn sklearn',
    url='https://github.com/tgcsaba/KSig',
    packages=['ksig'],
    install_requires=['numpy', 'scikit-learn', 'cupy'],
    python_requires='>=3.7',
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
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
