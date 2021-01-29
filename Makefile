all: mc3 polychord

mc3:
	@cd ./modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"

polychord:
	@cd ./modules/PolyChordLite && python setup.py --no-mpi install
	@echo "Finished installing pypolychord.\n"

