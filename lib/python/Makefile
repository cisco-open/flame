all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/fledge*.whl

clean:
	@rm -rf build dist fledge.egg-info
