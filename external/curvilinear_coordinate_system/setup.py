from setuptools import setup, find_packages

setup(
    name='commonroad-curvilinear-coordinate-system',  
    version='2019.1',
    description='Curvilinear coordinate system for CommonRoad scenarios.',
    url='https://commonroad.in.tum.de/',
    author='Technical University of Munich',  
    author_email='commonroad-i06@in.tum.de',
    license='BSD',
    packages=['.', 'commonroad_ccosy'],
    include_package_data=True,
	install_requires=[
        'commonroad-io',
		'numpy>=1.13',
		'matplotlib>=2.2.2',
		'networkx',
	],
	extras_require={
		'doc':	['sphinx>=1.3.6',
				 'graphviz>=0.3',
				 'sphinx-autodoc-typehints>=1.3.0',
                 'sphinx_rtd_theme>=0.4.1',
                 'sphinx-gallery>=0.2.0',
                 'ipython>=6.5.0'],
	},
    data_files=[('.',['LICENSE'])],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
	    "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",	
    ],
    zip_safe=False,
)
