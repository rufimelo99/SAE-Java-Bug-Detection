from setuptools import find_packages, setup


def read_requirements():
    """Read the requirements.txt file and return its contents as a list."""
    requirements_path = "requirements.txt"
    with open(requirements_path, "r") as f:
        return [line.strip() for line in f.readlines()]


setup(
    name="drl_patches",
    version="0.0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            # 'your_command=your_module.your_module:main',
        ],
    },
    author="Rui Melo, Cláudia Mamede",
    author_email="rufimelo99@gmail.com",
    description="PhD work @ FEUP, PORTO",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
