#include "model.h"

void	model_free(t_model *model)
{
	if (!model)
		return ;
	if (model->layers)
	{
		for (size_t i = 0; i < model->layer_count; i++)
		{
			free(model->layers[i].w);
			free(model->layers[i].b);
		}
		free(model->layers);
	}
	free(model);
}

t_model	*model_create(size_t *shape, size_t size)
{
	t_model	*new_model;

	new_model = calloc(1, sizeof(t_model));
	if (!new_model)
		return (NULL);
	new_model->layer_count = size - 1;
	new_model->layers = calloc(new_model->layer_count, sizeof(t_layer));
	if (!new_model->layers)
		return(model_free(new_model), NULL);
	for (size_t i = 0; i < new_model->layer_count; i++)
	{
		new_model->layers[i].l_inputs = shape[i];
		new_model->layers[i].l_neurons = shape[i + 1];
		new_model->layers[i].w = calloc(shape[i] * shape[i + 1], sizeof(double));
		new_model->layers[i].b = calloc(shape[i + 1], sizeof(double));
		if (!new_model->layers[i].w || !new_model->layers[i].b)
			return (model_free(new_model), NULL);
	}
	return (new_model);
}

void	model_apply_activation(t_model *model, size_t layer, t_activation act)
{
	model->layers[layer].act = act;
}

void	model_init_random(t_model *model)
{
	double range = 0;

	if (!model)
		return ;
	for (size_t i = 0; i < model->layer_count; i++)
	{
		range = sqrt(6.0 / (double)(model->layers[i].l_neurons + model->layers[i].l_inputs));
		for (size_t j = 0; j < model->layers[i].l_neurons; j++)
		{
			for (size_t k = 0; k < model->layers[i].l_inputs; k++)
			{
				model->layers[i].w[j * model->layers[i].l_inputs + k] = 
					(((double)rand() / RAND_MAX) * 2.0 * range) - range;
			}
			model->layers[i].b[j] = 0.0;
		}
	}
}
