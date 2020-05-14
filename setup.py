import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="video_stabilizer",
    version="0.1",
    author="Rickard Sjoegren",
    author_email="rickard.sjoegren@sartorius.com",
    description="A minimal package for video stabilization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sartorius-research/video_stabilizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'opencv-python>=4.2.0',
    ],
    python_requires='>=3.6',
)