#include "activation.h"

static double	max(double *array, size_t size)
{
	double	m = array[0];

	for (size_t i = 0; i < size; i++)
	{
		if (m < array[i])
			m = array[i];
	}
	return (m);
}

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

static void	softmax(double *z, double *out, size_t size)
{
	double	sum = 0;
	double	max_z = max(z, size);

	for (size_t j = 0; j < size; j++)
		sum += exp(z[j] - max_z);

	for (size_t i = 0; i < size; i++)
		out[i] = exp(z[i] - max_z) / sum;
}

static void	backward_softmax(double *z, double *incoming, double *out, double *delta, size_t size)
{
	double avg_blame = 0;

	for (size_t i = 0; i < size; i++)
		avg_blame += incoming[i] * out[i];

	for (size_t i = 0; i < size; i++)
		delta[i] = out[i] * (incoming[i] - avg_blame);

	(void)z;
}

static const t_act_funs	act_funs[ACT_COUNT] = 
{
	[ACT_SIGMOID] = {"sigmoid", sigmoid, backward_sigmoid},
	[ACT_SOFTMAX] = {"softmax", softmax, backward_softmax}
};

const t_act_funs	*activation_get(t_activation activation)
{
	if (activation < ACT_COUNT)
		return(act_funs + activation);
	return (NULL);
}

int	activation_from_name(const char *name, t_activation *out)
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
