
import sys,os
from setuptools import setup, find_packages




with open('README.md') as readme_file:
    readme = readme_file.read()


def read(name):
    with open(os.path.join(this, name)) as f:
        return f.read()

this = os.path.dirname(os.path.realpath(__file__))
__description__ ='None'
__version__ = '0.1.0'
__copyright__ = ''
setup(name = "esFCM",

      version = __version__,
      description = __description__,
      long_description = __description__,
      author = 'Bahram Jafrasteh',
      author_email = 'b.jafrasteh@gmail.com',
      packages = find_packages(),
      license = __copyright__ + \
                ", Licensed under the MIT",
          classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English'
       
    ],
    install_requires=read('requirements.txt'),
    include_package_data=True,
    zip_safe=True,
)




