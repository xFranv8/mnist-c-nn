CC = gcc
CFLAGS = -Wall -O2 -Iinclude
LDFLAGS = -lm

SRC = src/main.c src/neural_net.c src/matrix.c data/mnist_loader.c
TARGET = mnist_nn

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)