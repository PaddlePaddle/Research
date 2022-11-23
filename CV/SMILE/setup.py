from setuptools import setup, find_packages
import paddletransfer

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setup(
    name = "paddletransfer",
    version = paddletransfer.__version__,
    author = "Baidu-BDL",
    author_email = "autodl@baidu.com",
    description = "transfer learning toolkits for finetune deep learning models",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9'
    ],
    packages = find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'numpy'
    ],
    license = 'Apache 2.0',
    keywords = "transfer learning toolkits for paddle models"
)
