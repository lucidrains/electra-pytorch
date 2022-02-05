from setuptools import setup, find_packages

setup(
  name = 'electra-pytorch',
  packages = find_packages(),
  version = '0.1.2',
  license='MIT',
  description = 'Electra - Pytorch',
  author = 'Erik Nijkamp, Phil Wang',
  author_email = 'erik.nijkamp@gmail.com, lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/electra-pytorch',
  keywords = [
    'transformers',
    'artificial intelligence',
    'pretraining'
  ],
  install_requires=[
    'torch>=1.6.0',
    'transformers==3.0.2',
    'scipy',
    'sklearn'
  ],
  setup_requires=[
    'pytest-runner'
  ],
  tests_require=[
    'pytest',
    'reformer-pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)