PYFILES=$(wildcard *.py)
OPTS_PYLINT=--rcfile ./.pylintrc -j 0 --extension-pkg-whitelist=cv2,mpi4py --disable=W0311,W0718,C0103,C0114,C0116,C0301,C0415
#OPTS_MYPY=--disable-error-code=import-untyped
OPTS_EGREP_FILTER="cpu_feature_guard|Unable to register|rebuild TensorFlow"

default:
	@ echo "run 'make <test|lint|clean>'.."

test:
	@ echo "TESTING.."
	@ ($(foreach PYFILE,$(filter-out nbmerge.py,$(PYFILES)), echo "** TEST  $(PYFILE)" && (./$(PYFILE) 2>&1 | sed 's/^/    /g') &&) true) | egrep -v  $(OPTS_EGREP_FILTER)
	@ echo "** TEST  nbmerge.py" && nbmerge.py ../L01/intro.ipynb ../L02/performance_metrics.ipynb ../L03/supergruppe_diskussion.ipynb -o testmerge.ipynb
	@ echo "DONE"

lint:
	@ echo "LINTING.."
	@ $(foreach PYFILE, $(PYFILES),echo CHECK  $(PYFILE) && pylint $(OPTS_PYLINT) $(PYFILE) ;)
	@ echo "DONE"

F=isFloat
isFunctionUsed:
	@ #grep $F * -R
	@ echo "LOOKING for '$F' in files.."
	find ../ -iname '*.ipynb' -exec grep -H -n $F {} \;
	find ../ -iname '*.py'    -exec grep -H -n $F {} \;

edit:
	@ joe Makefile	

clean:
	@ rm -rf testmerge.ipynb __pycache__/ Utils/__pycache__/ .mypy_cache/
