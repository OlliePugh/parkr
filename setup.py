import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parkr",
    version="1.0.0",
    author="Ollie Pugh",
    author_email="oliver.pugh@mac.com",
    description="A lightweight basic neural networks library in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OlliePugh/parkr",
    project_urls={
        "Bug Tracker": "https://github.com/OlliePugh/parkr/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ]
    
)