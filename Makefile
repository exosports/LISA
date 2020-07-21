all: mc3

mc3:
	@cd ./modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"

