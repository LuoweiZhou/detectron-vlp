import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add lib to PYTHONPATH
# lib_path = '/private/home/xinleic/pyramid/lib'
lib_path = '/private/home/luoweizhou/subsystem/detectron-xinlei/lib'

print 'current lib path: %s' % lib_path
add_path(lib_path)
