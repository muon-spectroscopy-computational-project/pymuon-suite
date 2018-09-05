# Get required external dependency
dep/parsefmt.zip:
	mkdir -p dep
	wget https://github.com/CCP-NC/parse-fmt/archive/v0.5.zip -O dep/parsefmt.zip

install:
	pip install ./

install-all: dep/parsefmt.zip
	pip install dep/parsefmt.zip
	pip install ./