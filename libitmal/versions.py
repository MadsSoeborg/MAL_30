#!/usr/bin/env python3

import importlib
import sys

def Versions():
	def PrintVer(libname, version):
		print(f"{str(libname) + ' version:':28s} {version}")

	def TryPrintVersion(libname):
		try:
			lib = importlib.import_module(libname)
			if "__version__" in dir(lib):
				PrintVer(libname, f"{lib.__version__}")
			else:
				print(f"WARNING: library '{libname}' does not have a '__version__' attribute", file=sys.stderr)
		except ModuleNotFoundError:
			print(f"WARNING: could not find library '{libname}' in path", file=sys.stderr)

	PrintVer("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

	TryPrintVersion("numpy")
	TryPrintVersion("sklearn")
	TryPrintVersion("keras")
	TryPrintVersion("tensorflow")
	TryPrintVersion("tensorflow.keras")
	TryPrintVersion("cv2")
	TryPrintVersion("pytorch")

def TestAll():
	print("Test versions..")
	Versions()
	print("ALL OK")

if __name__ == '__main__':
	TestAll()
