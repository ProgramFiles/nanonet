PACKAGE  ?= ont-basecaller-ann
MAJOR    ?= 0
MINOR    ?= 0
SUB      ?= 1
PATCH    ?= 1
CODENAME ?= $(shell lsb_release -cs)
SEDI      = sed -i

.PHONY: install develop clean


install:
	python setup.py install

develop: clean
	python setup.py develop --user

deb: clean
	python setup.py --command-packages=stdeb.command bdist_deb

clean:
	python setup.py clean
	rm -rf build/ dist/ deb_dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete

