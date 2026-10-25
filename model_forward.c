#include "model.h"

double	predict_neuron(double *weights, double bias, double *input, size_t size)
{
	double	z = 0;

	for(size_t i = 0; i < size; i++)
		z += weights[i] * input[i];
	return (z += bias);
}

double	*forward(t_model *model, double *input)
{
	double	*current_input = input;
	double	*layer_output = NULL;
	const t_act_funs	*layer_act = NULL;

	for (size_t i = 0; i < model->layer_count; i++)
	{
		layer_output = calloc(model->layers[i].l_neurons, sizeof(double));
		if (!layer_output)
		{
			if (current_input != input)
				free(current_input);
			return (NULL);
		}
		for (size_t j = 0; j < model->layers[i].l_neurons; j++)
		{
			layer_output[j] = predict_neuron(
				&model->layers[i].w[j * model->layers[i].l_inputs],
				model->layers[i].b[j], current_input, model->layers[i].l_inputs);
		}
		layer_act = get_activation(model->layers[i].act);
		if (!layer_act)
		{
			if (current_input != input)
				free(current_input);
			return (NULL);
		}
		layer_act->forward_fun(layer_output, layer_output, model->layers[i].l_neurons);
		if (current_input != input)
			free(current_input);
		current_input = layer_output;
	}
	return (layer_output);
}
