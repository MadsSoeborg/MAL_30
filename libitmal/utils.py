#!/usr/bin/env python3

import random
import contextlib as ctxlib
from math import inf, nan
from collections.abc import Iterable # fixes Iterable and abc namespace error
import inspect # NOTE: for VarName
import re      # NOTE: for VarName
import numpy as np

def isList(x):
	#NOTE: should use python types instead of cmp with string!
	#return str(type(x))=="<class 'list'>"
	return isinstance(x, list)

def isNumpyArray(x):
	#NOTE: should use python types instead of cmp with string!
	#return str(type(x))=="<class 'numpy.ndarray'>"
	return isinstance(x, np.ndarray)

def isFloat(x):
	# is there a python single/double float??
	return isinstance(x,  (float, np.float32, np.float64)) # NOTE:  np.float128 not present on Windows

# Checks that a 'float' is 'sane'
def CheckFloat(x, checkrange=False, xmin=1E-200, xmax=1E200, verbose=0):
	if verbose>1:
		print(f"CheckFloat({x}, type={type(x)}")
	if isinstance(x, Iterable):
		for i in x:
			CheckFloat(i, checkrange=checkrange, xmin=xmin, xmax=xmax, verbose=verbose)
	else:
		#if (isinstance(x,int)):
		#    print("you gave me an integer, that was ignored")
		#    return
		assert isFloat(x),  f"x={x} is not a float/float64/numpy.float32/64/128, but a type={type(x)}"
		#assert np.isnan(x)  is False , f"x={x} is NAN"
		assert x==x, "x is NAN"
		#assert np.isinf(x)  is False, "x is inf"
		#assert np.isinf(-x) is False, "x is -inf"
		# NOTE: missing test for denormalized float
		if checkrange:
			z=np.fabs(x)
			assert z>=xmin, f"abs(x)={z} is smaller that expected min value={xmin}"
			assert z<=xmax, f"abs(x)={z} is larger that expected max value={xmax}"
		if verbose>0:
			print(f"CheckFloat({x}, type={x} => OK")

# Checks that two 'floats' are 'close'
def AssertInRange(x, expected, eps=1E-9, autoconverttofloat=True, verbose=0):
	# NOTE: alternative approach is to use numpy.isclose()
	if isinstance(x, Iterable):
		if isinstance(expected, Iterable):
			n=len(x)
			for i in range(n):
				if verbose>2:
					print(f"range: x[{i}]={x[i]}")
				AssertInRange(x[i], expected[i], eps, autoconverttofloat, verbose)
		else:
			norm = np.linalg.norm(x)
			if verbose>2:
				print("norm=",norm)
			AssertInRange(norm, expected, eps, autoconverttofloat, verbose)
	else:
		assert eps>=0, "eps is less than zero"
		if autoconverttofloat and (not isFloat(x) or not isFloat(expected) or not isFloat(eps)):
			if verbose>1:
				print(f"notice: autoconverting x={x} to float..")
			AssertInRange(1.0*x, 1.0*expected, 1.0*eps, False, verbose)
		else:
			CheckFloat(x)
			CheckFloat(expected)
			CheckFloat(eps)
			x0 = expected - eps
			x1 = expected + eps
			ok = x0 <= x <= x1
			absdiff = np.fabs(x-expected)
			if verbose > 0:
				print(f"CheckInRange(x={x}, expected={expected}, eps={eps}: x in [{x0}; {x1}] => {ok}")
			assert ok, f"x={x} is not within the range [{x0}; {x1}] for eps={eps}, got eps={absdiff}"

def InRange(x, expected ,eps=1E-9, verbose=0):
	try:
		AssertInRange(x, expected, eps, True, verbose)
		return True
	except Exception:
		return False

def ResetRandom(the_seed=1):
	# reset random
	random.seed(the_seed)
	np.random.seed(the_seed)

def VarName(x): # NOTE: rather hacky way to get some dbg info
	frame = inspect.currentframe().f_back
	s = inspect.getframeinfo(frame).code_context[0]
	r = re.search(r"\((.*)\)", s).group(1)
	#print("{} = {}".format(r,x))
	assert x is not None
	assert r is not None
	assert not r==""
	return r

def PrintMatrix(X, label="", precision=2, threshold=100, edgeitems=1, linewidth=80, suppress=True):
	@ctxlib.contextmanager
	def printoptions(*args, **kwargs):
		original = np.get_printoptions()
		np.set_printoptions(*args, **kwargs)
		try:
			yield
		finally:
			np.set_printoptions(**original)

	s = "".ljust(len(label))
	if label != "":
		print(label, end='')

	if threshold <= 0:
		threshold = X.size + 1

	with printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth, suppress=suppress):
		t = str(X).replace("\n","\n"+s)
		print(t)

######################################################################################################
#
# TESTS
#
######################################################################################################

def TEST(expr):
	# NOTE: test isjust a simple assert for now
	assert expr, "TEST FAILED"

def TestCheckFloat():
	e=0
	CheckFloat(42.)
	try:
		CheckFloat(42)
	except Exception:
		e += 1

	assert e==1, f"Test of CheckFloat(int) failed, expected e=1 but got {e}"
	e = 0

	z=nan
	try:
		CheckFloat(z)
	except Exception:
		e += 1

	assert e==1,f"Test of CheckFloat(nan) failed, expected e=1 but got {e}"
	e = 0

	try:
		CheckFloat(inf)
	except Exception:
		e += 1
	try:
		CheckFloat(-inf)
	except Exception:
		e += 1

	#assert e==2,f"Test of CheckFloat(inf/-inf) failed, expected e=2 but got {e}"
	e = 0

	try:
		CheckFloat(20.,True,1E-3,19.9)
	except Exception:
		e += 1

	assert e==1,f"Test of CheckFloat(tuple) failed, expected e=1 but got {e}"
	e = 0

	AssertInRange([1, 2, 3], [1, 2, 3.1], .2)
	try:
			AssertInRange([1, 2, 3], [1, 2, 3.1], .01)
	except Exception:
		e += 1

	assert e==1,f"Test of CheckFloat(inrange) failed, expected e=1 but got {e}"
	e = 0

	print("TEST: OK")

def TestVarName():
	spam = 42
	v=VarName(spam)
	TEST(v=="spam")

def TestPrintMatrix():
	print("TestPrintMatrix...(no regression testing)")
	X = np.matrix([[1,2],[3.0001,-100],[1,-1]])

	PrintMatrix(X,"X=",precision=1)
	PrintMatrix(X,"X=",precision=10,threshold=2)
	PrintMatrix(X,"X=",precision=10,edgeitems=0,linewidth=4)
	PrintMatrix(X,"X=",suppress=False)
	print("OK")

def TestAll():
	TestPrintMatrix()
	TestCheckFloat()
	TestVarName()
	print("ALL OK")

if __name__ == '__main__':
	TestAll()
