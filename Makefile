all: mc3 polychord

mc3:
	@cd ./lisa/modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"

polychord:
	@cd ./lisa/modules/PolyChordLite && python setup.py --no-mpi install
	@echo "Finished installing pypolychord.\n"

