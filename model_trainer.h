#ifndef MODEL_TRAINER_H
# define MODEL_TRAINER_H

# include "model.h"

typedef enum	e_loss
{
	LOSS_BCE,
	LOSS_CROSS_ENTROPY,
	LOSS_COUNT,
}				t_loss;

typedef struct	s_layer_data
{
	double	*z;
	double	*out;
	double	*delta;
}				t_layer_data;

typedef struct	s_trainer
{
	t_model			*model;
	double			*input;
	t_layer_data	*l_data;
	double			*incoming;
	t_loss			loss;
}				t_trainer;

void		trainer_free(t_trainer *trainer);
t_trainer	*trainer_create(t_model *model, t_loss loss);
int			trainer_forward(t_trainer *trainer, double *input);
int			trainer_backward(t_trainer *trainer, double *target);
void		trainer_update(t_trainer *trainer, double lr);
double		*trainer_output(t_trainer *trainer);
double		trainer_loss(t_trainer *trainer, double *target);

#endif
