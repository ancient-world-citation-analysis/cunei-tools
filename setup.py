from setuptools import setup, find_packages

setup(
    name="cunei-tools",
    version="0.1.0",
    description="Cuneiform NLP utilities: word segmentation and script conversion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="CJ Thompson, Adam Anderson",
    author_email="",
    url="https://github.com/ancient-world-citation-analysis/ElamiteDatasetLab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "convert": ["pandas"],
        "all": ["pandas"],
    },
    entry_points={
        "console_scripts": [
            "cunei-seg=cunei_tools.cli:seg_main",
            "cunei-conv=cunei_tools.cli:conv_main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="cuneiform nlp segmentation akkadian sumerian elamite",
)
