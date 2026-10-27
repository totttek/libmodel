#ifndef	ACTIVATION_H
# define ACTIVATION_H

# include <stdlib.h>
# include <string.h>
# include <math.h>

typedef enum
{
	ACT_SIGMOID,
	ACT_COUNT
}	t_activations;

typedef void	(*t_act_forward)(double *z, double *out, size_t size);
typedef void	(*t_act_backward)(double *z, double *incoming, double *out, double *delta, size_t size);

typedef struct	s_act_funs
{
	const char		*name;
	t_act_forward	forward_fun;
	t_act_backward	backward_fun;
}				t_act_funs;

const t_act_funs	*activation_get(t_activations activation);
int	activation_from_name(const char *name, t_activations *out);

#endif
