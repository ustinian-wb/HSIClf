from distutils.core import setup, Extension
import shutil

module = Extension('ERSModule',
                    sources = ['erspy.cpp', 'MERCCInput.cpp', 'MERCOutput.cpp', 'MERCDisjointSet.cpp', 'MERCFunctions.cpp', 'MERCLazyGreedy.cpp'])

setup(name = 'PackageName', 
      version = '1.0',
      description = 'This is a Python wrapper for ERS',
      ext_modules = [module])
      
#copy2 your path
shutil.copy2('build\lib.win-amd64-3.9\ERSModule.cp39-win_amd64.pyd', 'ERSModule.pyd')
