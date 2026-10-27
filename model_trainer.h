#ifndef MODEL_TRAINER_H
# define MODEL_TRAINER_H

# include "model.h"

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
}				t_trainer;

void		trainer_free(t_trainer *trainer);
t_trainer	*trainer_create(t_model *model);
void		model_init_random(t_model *model, double range);
int			trainer_forward(t_trainer *trainer, double *input);
int			trainer_backward(t_trainer *trainer, double *target);
void	trainer_update(t_trainer *trainer, double lr);

#endif
