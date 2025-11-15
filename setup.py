from setuptools import setup, find_packages

setup(
    name='my-python-project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Aquí puedes listar las dependencias del proyecto
    ],
    author='Tu Nombre',
    author_email='tu.email@example.com',
    description='Descripción de tu proyecto',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tu_usuario/my-python-project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)