from setuptools import setup, find_packages

setup(name='johannsen',
      version='0.0.1',
      description='Python implementation of the Johansen test for cointegration',
      author='Isaiah Yoo',
      author_email='',
      maintainer='Isaiah Yoo',
      maintainer_email='',
      packages=find_packages(),
      install_requires=[
          'scipy>=0.18.0',
          'statsmodels>=0.6.1',
          'numpy>=1.11.1',
      ],
      include_package_data=True,
)
