#ifndef MODEL_H
# define MODEL_H

# include "activation.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

# define MODEL_FORMAT_NAME "libmodel v"
# define MODEL_FORMAT_VERSION 2

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
	t_activation act;
}				t_layer;

typedef struct	s_model
{
	t_layer	*layers;
	size_t	layer_count;
}				t_model;

/* model.c - allocation */
t_model	*model_create(size_t *shape, size_t size);
void	model_free(t_model *model);

/* model_io.c - loading from a model file (shape line + one row per neuron) */
t_model	*model_load(char *filename);
int		model_save(t_model *model, char *filename);

void	model_apply_activation(t_model *model, size_t layer, t_activation act);
void	model_init_random(t_model *model);

/* model_forward.c - inference */
double	model_neuron_z(double *weights, double bias, double *input, size_t size);
double	*model_forward(t_model *model, double *input);

#endif
