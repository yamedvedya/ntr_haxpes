#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys

# ----------------------------------------------------------------------

ui_compilers = {"linux": {
    "pyqt": "pyuic",
    "pyside": "pyside-uic"
},
    "windows": {
        "pyqt": "C://Users//matveyev//AppData//Local//Programs//Python//Python37-32//Scripts//pyuic5.exe",
        "pyside": ""
    }
}

# ----------------------------------------------------------------------
def compile_uis(ui_compiler, in_dir, out_dir):
    """
    """
    for f in [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))
              and os.path.splitext(f)[-1] == ".ui"]:        # simplify this loop TODO
        base, ext = os.path.splitext(f)
        post, comp = ("_ui", ui_compiler)

        cmd = "{} {}/{} -o {}/{}{}.py".format(comp, in_dir, f, out_dir, base, post)
        print(cmd)
        os.system(cmd)

# ----------------------------------------------------------------------
if __name__ == "__main__":

    code_dir = os.path.dirname(sys.argv[0])
    src_dir = code_dir + '/src/widgets'
    uis_dir = code_dir + "/src/ui"

    print("Making widgets folder...")

    if not os.path.isdir('src/widgets'):
        os.mkdir('src/widgets')
        f = open('src/widgets/__init__.py', 'tw', encoding='utf-8')
        f.close()

    print("Removing pyc files...")

    for root, dirs, files in os.walk(src_dir):
        for f in [f for f in files if f.endswith(".pyc")]:
            if sys.platform == "linux" or sys.platform == "linux2":
                os.system("rm {}".format(os.path.join(root, f)))
            elif sys.platform == "win32":
                os.remove(os.path.join(root, f))

    print("Removing uis and rcs...")
    for root, dirs, files in os.walk(uis_dir):
        for f in [f for f in files if (f.endswith(".pyc") or f.endswith(".py"))
                                      and f != "__init__.py"]:
            if sys.platform == "linux" or sys.platform == "linux2":
                os.system("rm {}".format(os.path.join(root, f)))
            elif sys.platform == "win32":
                os.remove(os.path.join(root, f))

    print("All removed!")

    if sys.platform == "linux" or sys.platform == "linux2":
        lib_name, sys_name = "pyqt", "linux"
    elif sys.platform == "win32":
        lib_name, sys_name = "pyqt", "windows"

    compile_uis(ui_compilers[sys_name][lib_name], uis_dir, src_dir)

    print("All OK!")