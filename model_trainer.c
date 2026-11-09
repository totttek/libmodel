#include "model_trainer.h"

void	trainer_free(t_trainer *trainer)
{
	if (!trainer)
		return ;
	if (trainer->l_data)
	{
		for (size_t i = 0; i < trainer->model->layer_count; i++)
		{
			free(trainer->l_data[i].delta);
			free(trainer->l_data[i].out);
			free(trainer->l_data[i].z);
		}
		free(trainer->l_data);
	}
	free(trainer->incoming);
	free(trainer);
}

t_trainer	*trainer_create(t_model *model)
{
	t_trainer	*trainer = calloc(1, sizeof(t_trainer));
	size_t		max_neurons = 0;

	if (!trainer)
		return(NULL);
	trainer->model = model;
	trainer->l_data = calloc(model->layer_count, sizeof(t_layer_data));
	if (!trainer->l_data)
		return(trainer_free(trainer), NULL);
	for (size_t i = 0; i < model->layer_count; i++)
	{
		trainer->l_data[i].z = calloc(model->layers[i].l_neurons, sizeof(double));
		trainer->l_data[i].out = calloc(model->layers[i].l_neurons, sizeof(double));
		trainer->l_data[i].delta = calloc(model->layers[i].l_neurons, sizeof(double));
		if (!trainer->l_data[i].z || !trainer->l_data[i].out || !trainer->l_data[i].delta)
			return(trainer_free(trainer), NULL);
		if (model->layers[i].l_neurons > max_neurons)
			max_neurons = model->layers[i].l_neurons;
	}
	trainer->incoming = calloc(max_neurons, sizeof(double));
	if (!trainer->incoming)
		return(trainer_free(trainer), NULL);
	return (trainer);
}

static double	*get_input(t_trainer *trainer, size_t layer_index)
{
	if (layer_index == 0)
		return (trainer->input);
	return(trainer->l_data[layer_index - 1].out);
}

int trainer_forward(t_trainer *trainer, double *input)
{
	const t_act_funs	*layer_act = NULL;
	t_layer				*layer = NULL;

	trainer->input = input;

	for (size_t i = 0; i < trainer->model->layer_count; i++)
	{
		layer = &trainer->model->layers[i];
		for (size_t j = 0; j < layer->l_neurons; j++)
		{
			trainer->l_data[i].z[j] = model_neuron_z(
				&layer->w[j * layer->l_inputs],
				layer->b[j], get_input(trainer, i), layer->l_inputs);
		}
		layer_act = activation_get(layer->act);
		if (!layer_act)
			return (-1);
		layer_act->forward_fun(trainer->l_data[i].z, trainer->l_data[i].out, layer->l_neurons);
	}
	return (0);
}

static void	compute_incoming(t_trainer *trainer, size_t k)
{
	for (size_t i = 0; i < trainer->model->layers[k].l_neurons; i++)
	{
		trainer->incoming[i] = 0;
		for (size_t j = 0; j < trainer->model->layers[k + 1].l_neurons; j++)
		{
			trainer->incoming[i] += trainer->l_data[k + 1].delta[j] * trainer->model->layers[k + 1].w[j * trainer->model->layers[k + 1].l_inputs + i];
		}
	}
}

int	trainer_backward(t_trainer *trainer, double *target)
{
	size_t			i = trainer->model->layer_count - 1;
	t_layer_data	*layer = &trainer->l_data[i];
	const t_act_funs		*act_fun = NULL;

	for (size_t j = 0; j < trainer->model->layers[i].l_neurons; j++)
	{
		layer->delta[j] = layer->out[j] - target[j];
	}
	for (size_t j = i; j-- > 0; )
	{
		layer = &trainer->l_data[j];
		compute_incoming(trainer, j);
		act_fun = activation_get(trainer->model->layers[j].act);
		if (!act_fun)
			return(-1);
		act_fun->backward_fun(layer->z, trainer->incoming, layer->out, layer->delta, trainer->model->layers[j].l_neurons);
	}
	return (0);
}

void	trainer_update(t_trainer *trainer, double lr)
{
	if (!trainer) return ;

	for (size_t i = 0; i < trainer->model->layer_count; i++)
	{
		for (size_t j = 0; j < trainer->model->layers[i].l_neurons; j++)
		{
			for (size_t k = 0; k < trainer->model->layers[i].l_inputs; k++)
			{
				trainer->model->layers[i].w[j * trainer->model->layers[i].l_inputs + k] -= lr * trainer->l_data[i].delta[j] * get_input(trainer, i)[k];
			}
			trainer->model->layers[i].b[j] -= lr * trainer->l_data[i].delta[j];
		}
	}
}

double	*trainer_output(t_trainer *trainer)
{
	return(trainer->l_data[trainer->model->layer_count - 1].out);
}

double	trainer_loss(t_trainer *trainer, double *target)
{
	size_t	neurons = trainer->model->layers[trainer->model->layer_count - 1].l_neurons;
	double	*p = NULL;
	double	pc = 0;
	double	final_loss = 0;

	p = trainer_output(trainer);
	for (size_t i = 0; i < neurons; i++)
	{
		pc = fmax(1e-7, fmin(p[i], 1 - 1e-7));
		final_loss -= (target[i] * log(pc) + (1 - target[i]) * log(1 - pc));
	}
	return (final_loss);
}
