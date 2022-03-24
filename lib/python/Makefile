all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/flame*.whl

uninstall:
	@pip3 uninstall -y flame

clean:
	@rm -rf build dist flame.egg-info
