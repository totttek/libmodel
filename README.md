# libmodel

A small C library for fully connected neural networks of any shape: creating, initialising, training, saving, loading and running them. It is built as a static library (`libmodel.a`) so several programs can share the same model code.

## Build

```sh
make            # builds libmodel.a
make clean      # removes object files
make fclean     # removes object files and the library
```

To use it from another program, include the header(s) and link the library (plus `-lm` for `exp`/`log`):

```sh
cc -I path/to/libmodel -c my_program.c
cc my_program.o -L path/to/libmodel -lmodel -lm -o my_program
```

Include `model.h` for inference only, or `model_trainer.h` (which includes `model.h`) to train.

## Files

| File | Contents |
|---|---|
| `model.h` | model structs, file format constants, model functions |
| `model.c` | `model_create`, `model_free`, `model_apply_activation`, `model_init_random` |
| `model_io.c` | `model_load`, `model_save` and the parsing helpers |
| `model_forward.c` | `model_neuron_z`, `model_forward` (inference) |
| `activation.h` / `activation.c` | the activation table: forward and backward function per activation |
| `model_trainer.h` / `model_trainer.c` | training buffers, forward/backward pass, update, losses |

## Data layout

A model is an array of layers. Each layer stores its weights as one flat `l_neurons × l_inputs` matrix: neuron `n`'s weights start at `w[n * l_inputs]`, followed by one bias per neuron in `b`, and the layer's activation.

```c
typedef struct s_layer { double *w; double *b; size_t l_inputs; size_t l_neurons; t_activation act; } t_layer;
typedef struct s_model { t_layer *layers; size_t layer_count; } t_model;
```

Training state lives separately in a `t_trainer`, so a model only ever holds its parameters. The trainer *borrows* the model and the current input (it never frees them) and *owns* per-layer `z`, `out` and `delta` buffers plus one scratch buffer for incoming gradients, all allocated once in `trainer_create`.

## API: models

```c
t_model *model_create(size_t *shape, size_t size);
```
Allocates a model with every weight and bias set to 0 and every activation set to sigmoid. `shape` lists the layer widths, input first (e.g. `{3, 2, 1}` gives 2 layers: 3→2 and 2→1). `size` is the number of entries in `shape` and must be at least 2. Returns `NULL` on allocation failure.

```c
void model_apply_activation(t_model *model, size_t layer, t_activation act);
```
Sets one layer's activation (layer 0 is the first layer after the input), e.g. `model_apply_activation(model, 1, ACT_SOFTMAX)` for a softmax output on a 2-layer model.

```c
void model_init_random(t_model *model);
```
Xavier (Glorot) initialisation: each layer's weights are drawn uniformly from `[-limit, limit]` with `limit = sqrt(6 / (l_inputs + l_neurons))`, computed per layer, and every bias is set to 0. This keeps sigmoid/softmax inputs out of saturation even for wide layers like MNIST's 784 inputs. Uses `rand()`; seed it yourself with `srand` for reproducible runs.

```c
t_model *model_load(char *filename);
int      model_save(t_model *model, char *filename);
```
Read / write a model file (format below). `model_load` returns `NULL` if the file can't be opened, has the wrong format or version, or doesn't match its own shape line. `model_save` returns 0 on success, -1 on failure (including a failed `fclose`). A model saved and loaded again is identical, since weights are written with `%.17g`.

```c
double *model_forward(t_model *model, double *input);
```
Runs the network on `input` (as many values as the first shape entry) and returns a newly allocated array with the last layer's outputs. The caller frees it. Returns `NULL` on allocation failure or an invalid activation.

```c
double model_neuron_z(double *weights, double bias, double *input, size_t size);
```
One neuron's raw score: `weights · input + bias` (the activation is applied per layer afterwards).

```c
void model_free(t_model *model);
```
Frees a model, including a partially built one. Accepts `NULL`.

## API: training

```c
t_trainer *trainer_create(t_model *model, t_loss loss);
void       trainer_free(t_trainer *trainer);
```
Allocate / free the training buffers for a model. Free the trainer before the model. `loss` is `LOSS_BCE` or `LOSS_CROSS_ENTROPY`; `trainer_create` returns `NULL` for an unknown loss or when the loss doesn't match the output layer's activation (see Losses below).

```c
int     trainer_forward(t_trainer *trainer, double *input);
double *trainer_output(t_trainer *trainer);
double  trainer_loss(t_trainer *trainer, double *target);
int     trainer_backward(t_trainer *trainer, double *target);
void    trainer_update(t_trainer *trainer, double lr);
```
One training step on one example, in this order: forward pass (keeps every layer's `z` and `out`), optional loss, backward pass (computes every layer's `delta`), then the weight update `w -= lr · delta · input`, `b -= lr · delta`. `target` has one value per output neuron. `trainer_forward` and `trainer_backward` return 0 on success and -1 on an invalid activation. `trainer_output` returns the last layer's outputs. `trainer_loss` returns the loss for the current example, or -1 for an unknown loss.

## Losses

The loss belongs to the trainer, not the model: a saved model can be loaded and run without one.

| Loss | Required output activation | Per-example loss |
|---|---|---|
| `LOSS_BCE` (binary cross-entropy) | sigmoid | `-Σ [y·log(p) + (1-y)·log(1-p)]` over all outputs |
| `LOSS_CROSS_ENTROPY` (categorical) | softmax | `-Σ y·log(p)`, i.e. `-log(p_correct)` for a one-hot target |

Both pairs are "matching" (canonical) pairs: the derivative of the loss cancels the derivative of the output activation, so the output delta is `out - target` in both cases. `trainer_backward` uses that fused delta directly, which is also numerically more stable than going through the activation's backward function. This is why `trainer_create` rejects mismatched pairs such as BCE on a softmax output. Probabilities are clamped to `[1e-7, 1 - 1e-7]` before taking logs.

## Activations

Each activation is one row in a table in `activation.c`: a name (used in model files), a forward function and a backward function. Layers store the enum value; `activation_get` returns the row (or `NULL` for an invalid value) and `activation_from_name` maps a file name back to the enum. Adding an activation means one new enum value before `ACT_COUNT` and one new table row.

Activation functions receive the whole layer (`z`, `out`, size), so an activation can depend on every neuron in the layer, as softmax does.

| Name | Forward | Backward |
|---|---|---|
| `sigmoid` | `1 / (1 + e^-z)` per neuron | `delta = incoming · out · (1 - out)` |
| `softmax` | `e^(z_i - max z) / Σ e^(z_j - max z)`; subtracting the max avoids overflow and doesn't change the result | `delta_i = out_i · (incoming_i - Σ_j incoming_j · out_j)` (the softmax Jacobian applied to the incoming gradient, without building the matrix) |

The backward function is only used for hidden layers; the output layer's delta comes from the loss (see Losses).

## Model file format (v2)

```
libmodel v2
3,2,1
sigmoid,sigmoid
w,w,w,b
w,w,w,b
w,w,b
```

- **Line 1:** the format identifier and version. Files with another prefix or version are rejected.
- **Line 2:** the shape, comma-separated layer widths (input first). Every value must be a positive integer.
- **Line 3:** one activation name per layer.
- **Then one line per neuron**, layer by layer: its `l_inputs` weights followed by its bias.
- Values are comma-separated with no spaces. The final line may omit the trailing newline.

## AI use

The library's code is my own. I used Claude (Anthropic's AI assistant) for:

- Explaining the underlying math (backpropagation, Xavier initialisation, the softmax and cross-entropy derivatives) and discussing design options, for example the fused loss approach
- Code reviews that pointed me toward bugs instead of fixing them
- Writing this README (the API documentation)
- A mechanical rename of two type names (`t_activations` → `t_activation`, `t_losses` → `t_loss`)
