.PHONY: install develop clean

deb:
	rm -rf deb_dist dist
	python setup.py --command-packages=stdeb.command sdist_dsc bdist_deb

install:
	python setup.py install

develop: clean
	python setup.py develop --user

clean:
	python setup.py clean
	rm -rf build/ dist/ *.egg-info/
	find . -name '*.pyc' -delete
	find . -name '*.so' -delete
