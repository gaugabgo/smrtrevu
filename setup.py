from setuptools import setup, find_packages

setup(
    name='revu',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'click',
        'spacy',
        'pandas',
        'nbib',
        'matplotlib',
        'gensim',
        'umap-learn',
        'hdbscan',
        'bertopic',
        'scikit-learn',
        'sentence-transformers',
        'rispy',
        'plotly',
        'scipy',
        'datamapplot',
        'seaborn',
        'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl',
        'pycountry',
        'python-louvain',
        'geopandas',
        'geodatasets',
    ],
    entry_points={
        'console_scripts': [
            'revu = revu.scripts.cli:cli',
        ],
    },
)
