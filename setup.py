from setuptools import setup, find_packages

setup(
    name="Expression2RGB",
    packages=find_packages(),
    install_requires=[
        'Click',
    ],
    entry_points="""
        [console_scripts]
        expression2Img=expression2Img:cli
        """

)