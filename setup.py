from setuptools import setup, find_packages

setup(
    name="Expression2RGB",
    packages=find_packages(),
    install_requires=[
        'Click',
        'anndata==0.7.6',
        'scanpy==1.7.2',
        'torch==1.5.1',
        'opencv-python'

    ],
    entry_points="""
        [console_scripts]
        expression2Img=expression2Img:cli
        """

)