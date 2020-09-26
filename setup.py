
from setuptools import setup, find_packages

setup(
	name='OpenCLEnvironmentProvider',
	version='0.0.1',
	author='Daniel Hedman',
	author_email='daniel.hedman@ltu.se',
	url='https://github.com/Dankomaister/OpenCLEnvironmentProvider',
	packages=find_packages(),
	package_data={'': ['neighbor_list_kernel.cl']},
	include_package_data=True,
	python_requires='>=3.6',
	install_requires=[
		'pyopencl',
		'numpy',
		'schnetpack'
	],
	license='MIT',
	description='OpenCL environment provider for SchNetPack'
)
