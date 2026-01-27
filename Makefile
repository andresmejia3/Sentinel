# Makefile

.PHONY: build run clean

# The 'build' target compiles the binary to the root
build:
	go build -o sentinel ./cmd/sentinel/main.go

# The 'run' target builds and then executes it
run: build
	sentinel scan -i samples/3.mp4

# Removes the binary
clean:
	rm -f sentinel

sample:
	make build
	make run
