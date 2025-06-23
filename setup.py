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
        'pymupdf',
        'plotly'
        # add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'revu = revu.scripts.cli:cli',
        ],
    },
)
