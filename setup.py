from setuptools import setup

setup(name='gathering_mae',
      version='0.01',
      description='Multi agent gathering env',
      entry_points={
          'console_scripts': [
              'liftoff=liftoff.cmd:launch',
              'liftoff-prepare=liftoff.cmd:prepare',
              'liftoff-status=liftoff.cmd:status',
              'liftoff-abort=liftoff.cmd:abort',
          ],
      },
      url='https://github.com/andreicnica/gathering_mae',
      author='Andrei Nica',
      author_email='andreic.nica@gmail.com',
      license='MIT',
      packages=['gathering_mae'],
      install_requires=[
      ],
      zip_safe=False)
