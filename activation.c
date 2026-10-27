#include "activation.h"

static void	sigmoid(double *z, double *out, size_t size)
{
	for (size_t i = 0; i < size; i++)
		out[i] = 1.0 / (1.0 + exp(-z[i]));
}

static void	backward_sigmoid(double *z, double *incoming, double *out, double *delta, size_t size)
{
	for (size_t i = 0; i < size; i++)
		delta[i] = incoming[i] * out[i] * (1.0 - out[i]);

	(void)z;
}

static const t_act_funs	act_funs[ACT_COUNT] = 
{
	[ACT_SIGMOID] = { "sigmoid", sigmoid, backward_sigmoid}
};

const t_act_funs	*activation_get(t_activations activation)
{
	if (activation < ACT_COUNT)
		return(act_funs + activation);
	return (NULL);
}

int	activation_from_name(const char *name, t_activations *out)
{
	for (size_t i = 0; i < ACT_COUNT; i++)
	{
		if (!strcmp(name, act_funs[i].name))
		{
			*out = i;
			return (0);
		}
	}
	return (-1);
}
