
from setuptools import setup

setup(
	name='OpenCLEnvironmentProvider',
	version='0.0.1',
	author='Daniel Hedman',
	email='daniel.hedman@ltu.se',
	url='https://github.com/Dankomaister/OpenCLEnvironmentProvider',
	packages=['OpenCLEnvironmentProvider'],
	scripts=['OpenCLEnvironmentProvider/environment'],
	python_requires='>=3.6',
	install_requires=[
		'pyopencl',
		'numpy',
		'schnetpack'
	],
	license='MIT',
	description='OpenCL environment provider for SchNetPack'
)