#include "model.h"

void	free_model(t_model *model)
{
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

t_model	*create_model(size_t *shape, size_t size)
{
	t_model	*new_model;

	new_model = calloc(1, sizeof(t_model));
	if (!new_model)
		return (NULL);
	new_model->layer_count = size - 1;
	new_model->layers = calloc(new_model->layer_count, sizeof(t_layer));
	if (!new_model->layers)
		return(free_model(new_model), NULL);
	for (size_t i = 0; i < new_model->layer_count; i++)
	{
		new_model->layers[i].l_inputs = shape[i];
		new_model->layers[i].l_neurons = shape[i + 1];
		new_model->layers[i].w = calloc(shape[i] * shape[i + 1], sizeof(double));
		new_model->layers[i].b = calloc(shape[i + 1], sizeof(double));
		if (!new_model->layers[i].w || !new_model->layers[i].b)
			return (free_model(new_model), NULL);
	}
	return (new_model);
}
