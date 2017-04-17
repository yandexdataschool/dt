from __future__ import print_function

# Unfortunately, numpy should be installed before
# Also, installation requires fortran + openmp.
from numpy.distutils.core import Extension, setup

setup(
    name="decisiontrain",
    version='0.1',
    description="Decision Train - fastest gradient boosting in the world",
    long_description="""
        Decision Train is a modification of gradient boosting over decision trees.
        It is developed to process really lots of data using a single machine.
        """,
    url='https://arogozhnikov.github.io',

    # Author details
    author='Alex Rogozhnikov',
    author_email='axelr@yandex-team.ru',

    # Choose your license
    license='Apache 2.0',
    packages=['decisiontrain'],

    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    keywords='Machine Learning',

    ext_modules=[Extension(name='decisiontrain._dtrain', sources=['decisiontrain/_dtrain.f90'],
                           extra_link_args=["-lgomp"],
                           extra_f90_compile_args=["-fopenmp -O3 -march=native"])],

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'numpy >= 1.10',
        'scipy >= 0.15.0',
        'pandas >= 0.14.0',
        'scikit-learn >= 0.15.2',
        'six',
        'hep_ml == 0.4',
    ],
)
