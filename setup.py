from setuptools import setup, find_packages

setup(
    name = "parallelpandas",
    version = "0.1",
    packages = find_packages(),
    scripts = [],

    # metadata for upload to PyPI
    author = "Philipp Metzner",
    author_email = "philipp.metzner@gameduell.de",
    description = "parallel version of some pandas function",
    license = "MIT",
    keywords = "pandas mutliprocessing parallel",
    url = "https://github.com/GameDuell/parallelpandas",
)
