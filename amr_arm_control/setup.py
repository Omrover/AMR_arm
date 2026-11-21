from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_arm_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Omrover',
    maintainer_email='sanduich13@gmail.com',
    description='Cinemática y servidor de acciones del brazo de 4 GIM6010-8.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'arm_server = amr_arm_control.arm_server:main',
        'arm_controller = amr_arm_control.arm_controller:main',
        'arm_check = amr_arm_control.arm_check:main',
    ]},
)
