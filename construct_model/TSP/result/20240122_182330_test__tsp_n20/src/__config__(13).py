# This file is generated by numpy's setup.py
# It contains system_info results at the time of building this package.
__all__ = ["get_info","show"]


import os
import sys

extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')

if sys.platform == 'win32' and os.path.isdir(extra_dll_dir):
    if sys.version_info >= (3, 8):
        os.add_dll_directory(extra_dll_dir)
    else:
        os.environ.setdefault('PATH', '')
        os.environ['PATH'] += os.pathsep + extra_dll_dir

lapack_mkl_info={'libraries': ['blas', 'cblas', 'lapack', 'pthread', 'blas', 'cblas', 'lapack'], 'library_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/lib'], 'define_macros': [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)], 'include_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/include']}
lapack_opt_info={'libraries': ['blas', 'cblas', 'lapack', 'pthread', 'blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack'], 'library_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/lib'], 'define_macros': [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)], 'include_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/include']}
blas_mkl_info={'libraries': ['blas', 'cblas', 'lapack', 'pthread', 'blas', 'cblas', 'lapack'], 'library_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/lib'], 'define_macros': [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)], 'include_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/include']}
blas_opt_info={'libraries': ['blas', 'cblas', 'lapack', 'pthread', 'blas', 'cblas', 'lapack', 'blas', 'cblas', 'lapack'], 'library_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/lib'], 'define_macros': [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)], 'include_dirs': ['/home/pjy/anaconda3/envs/VRPDACT/include']}

def get_info(name):
    g = globals()
    return g.get(name, g.get(name + "_info", {}))

def show():
    for name,info_dict in globals().items():
        if name[0] == "_" or type(info_dict) is not type({}): continue
        print(name + ":")
        if not info_dict:
            print("  NOT AVAILABLE")
        for k,v in info_dict.items():
            v = str(v)
            if k == "sources" and len(v) > 200:
                v = v[:60] + " ...\n... " + v[-60:]
            print("    %s = %s" % (k,v))
