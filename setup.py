from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='gemmr',
      version='0.2.6',
      author='Markus Helmer',
      url='https://github.com/murraylab/gemmr',
      description='Generative Modeling of Multivariate Relationships',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
      ],
      python_requires='>=3.6',
      license='GPLv3',
      packages=find_packages(),
      install_requires=[
        'numpy>=1.20.1',
        'scipy',
        'pandas',
        'xarray==0.18.0',
        'netcdf4',
        'scikit-learn',
        'statsmodels',
        'joblib',
        'tqdm',
      ],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=True,
      include_package_data=True
)
