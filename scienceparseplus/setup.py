from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()
    required = [str(r) for r in required if r and not r.startswith("#")]

setup(
    name="scienceparseplus",
    version="0.0.1",
    description="PDF Extraction with Layout Analysis",
    package_dir={"": "src"},
    packages=find_packages("src"),
    long_description=readme,
    url="http://github.com/allenai/scienceparseplus",
    author="Zejiang Shen",
    author_email="shannons@allenai.org",
    keywords="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache 2.0",
    install_requires=required,
    extras_require={
        "dev": ["black==20.8b1", "pytest"],
    },
    python_requires=">=3.6",
    zip_safe=False,
)