.PHONY: install develop clean

install:
	python setup.py install

develop: clean
	python setup.py develop --user

deb: clean
	touch tmp
	rm -rf tmp *.deb
	python setup.py --command-packages=stdeb.command bdist_deb

clean:
	python setup.py clean
	rm -rf build/ dist/ deb_dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete
