# libmodel

A small C library for fully connected neural networks of any shape: allocation, loading from a model file, and the forward pass (inference). It is built as a static library (`libmodel.a`) so several programs can share the same model code.

## Build

```sh
make            # builds libmodel.a
make clean      # removes object files
make fclean     # removes object files and the library
```

To use it from another program, include the header and link the library (plus `-lm` for `exp`):

```sh
cc -I path/to/libmodel -c my_program.c
cc my_program.o -L path/to/libmodel -lmodel -lm -o my_program
```

## Files

| File | Contents |
|---|---|
| `model.h` | public structs and functions |
| `model.c` | `create_model`, `free_model` |
| `model_io.c` | `load_model` and its internal parsing helpers |
| `model_forward.c` | `predict_neuron`, `forward` |

## Data layout

A model is an array of layers. Each layer stores its weights as one flat `l_neurons × l_inputs` matrix: neuron `n`'s weights start at `w[n * l_inputs]`, followed by one bias per neuron in `b`.

```c
typedef struct s_layer { double *w; double *b; size_t l_inputs; size_t l_neurons; } t_layer;
typedef struct s_model { t_layer *layers; size_t layer_count; } t_model;
```

## API

```c
t_model *create_model(size_t *shape, size_t size);
```
Allocates a model with every weight and bias set to 0. `shape` lists the layer widths, input first (e.g. `{3, 2, 1}` gives 2 layers: 3→2 and 2→1). `size` is the number of entries in `shape` and must be at least 2. Returns `NULL` on allocation failure.

```c
t_model *load_model(char *filename);
```
Reads a model file and returns a filled model, or `NULL` if the file can't be opened, is malformed, or doesn't match its own shape line.

```c
double *forward(t_model *model, double *input);
```
Runs the network on `input` (which must have as many values as the first shape entry) and returns a newly allocated array with the last layer's outputs. The caller frees it. Returns `NULL` on allocation failure.

```c
double predict_neuron(double *weights, double bias, double *input, size_t size);
```
One neuron: `sigmoid(weights · input + bias)`.

```c
void free_model(t_model *model);
```
Frees a model, including a partially built one.

## Model file format

```
3,2,1
w,w,w,b
w,w,w,b
w,w,b
```

- **Line 1:** the shape, comma-separated layer widths (input first). Every value must be a positive integer.
- **Then one line per neuron**, layer by layer: its `l_inputs` weights followed by its bias.
- Values are comma-separated with no spaces. The final line may omit the trailing newline.
