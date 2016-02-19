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
