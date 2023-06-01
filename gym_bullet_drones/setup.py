from setuptools import setup, find_packages

setup(
    name='gym_bullet_drones',
    version='0.1',
    packages=find_packages(),
    description='A gym environment for multi-drone simulation based on PyBullet and ROS',
    author='Cao Jiawei',
    author_email='tslcaoj@nus.edu.sg',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)