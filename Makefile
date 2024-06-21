all: mc3 polychord pydream

mc3:
	@cd ./lisa/modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"

polychord:
	@cd ./lisa/modules/PolyChordLite && python setup.py --no-mpi install
	@echo "Finished installing pypolychord.\n"

pydream:
	@cd ./lisa/modules/PyDREAM && python setup.py install
	@echo "Finished installing PyDREAM.\n"
