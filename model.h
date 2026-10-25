#ifndef MODEL_H
# define MODEL_H

# include "activation.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

/*
** One fully connected layer.
** w is a flat l_neurons x l_inputs matrix: neuron n's weights start at
** w[n * l_inputs]. b holds one bias per neuron.
*/
typedef	struct	s_layer
{
	double *w;
	double *b;
	size_t l_inputs;
	size_t l_neurons;
	t_activations act;
}				t_layer;

typedef struct	s_model
{
	t_layer	*layers;
	size_t	layer_count;
}				t_model;

/* model.c - allocation */
t_model	*create_model(size_t *shape, size_t size);
void	free_model(t_model *model);

/* model_io.c - loading from a model file (shape line + one row per neuron) */
t_model	*load_model(char *filename);

/* model_forward.c - inference */
double	predict_neuron(double *weights, double bias, double *input, size_t size);
double	*forward(t_model *model, double *input);

#endif
