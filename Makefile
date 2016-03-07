.PHONY: install develop clean

install:
	python setup.py install

develop: clean
	python setup.py develop --user

deb: clean
	touch tmp
	rm -rf tmp *.deb
	python setup.py --command-packages=stdeb.command bdist_deb
#	mkdir tmp
#	dpkg -x deb_dist/*.deb tmp
#	dpkg-deb --control deb_dist/*.deb tmp/DEBIAN
#	(echo ; echo "update-alternatives --install /usr/bin/basecaller basecaller /usr/bin/nanonetcall") >> tmp/DEBIAN/postinst
#	(echo ; echo "update-alternatives --remove basecaller /usr/bin/nanonetcall") >> tmp/DEBIAN/prerm
#	dpkg -b tmp deb_dist/*.deb

clean:
	python setup.py clean
	rm -rf build/ dist/ deb_dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete
