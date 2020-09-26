import os
import io

from setuptools import setup, find_packages

def read(fname):
	with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
		return f.read()

setup(
	name='OpenCLEnvironmentProvider',
	version='0.0.1',
	author='Daniel Hedman',
	email='daniel.hedman@ltu.se',
	url='https://github.com/Dankomaister/OpenCLEnvironmentProvider',
	packages=find_packages('OpenCLEnvironmentProvider'),
	package_dir={'': 'OpenCLEnvironmentProvider'},
	python_requires='>=3.6',
	install_requires=[
		'pyopencl',
		'numpy',
		'schnetpack'
	],
	license='MIT',
	description='OpenCL environment provider for SchNetPack'
)