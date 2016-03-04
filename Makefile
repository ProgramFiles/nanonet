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

clean:
	python setup.py clean
	rm -rf build/ dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete

deb:
	touch tmp
	rm -rf tmp build *.deb
	python setup.py --command-packages=stdeb.command bdist_deb
	mkdir -p tmp/usr/bin
	cp -R deb-src/DEBIAN tmp/
	cp deb-src/control.t tmp/DEBIAN/control
	$(SEDI) "s/PACKAGE/$(PACKAGE)/g"   tmp/DEBIAN/control
	$(SEDI) "s/MAJOR/$(MAJOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/MINOR/$(MINOR)/g"       tmp/DEBIAN/control
	$(SEDI) "s/SUB/$(SUB)/g"           tmp/DEBIAN/control
	$(SEDI) "s/PATCH/$(PATCH)/g"       tmp/DEBIAN/control
	$(SEDI) "s/CODENAME/$(CODENAME)/g" tmp/DEBIAN/control
#	cp build/currennt tmp/usr/bin/
#	cp README LICENSE NOTICE tmp/usr/share/doc/ont-currennt
	(cd tmp; fakeroot dpkg -b . ../$(PACKAGE)-$(MAJOR).$(MINOR).$(SUB)-$(PATCH)~$(CODENAME).deb)
	rm -rf tmp
