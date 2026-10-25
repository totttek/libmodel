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

static size_t	count_commas(char *str)
{
	size_t	sep = 0;

	while (*str)
	{
		if (*str == ',')
			sep++;
		str++;
	}
	return (sep);
}

static int	parse_size(char	**p, size_t	*out)
{
	char	*endptr = NULL;
	size_t	tmp;

	tmp = strtoul(*p, &endptr, 10);
	if (!endptr || *p == endptr || (*endptr != ',' && *endptr != '\n') || tmp == 0)
		return (-1);
	*out = tmp;
	*p = endptr + 1;
	return (0);
}

static size_t	parse_shape(FILE *file, size_t **out)
{
	char	*first_line = NULL;
	char	*startptr = NULL;
	size_t	*shape = NULL;
	size_t	line_size = 0;
	size_t	count = 0;

	if (getline(&first_line, &line_size, file) <= 0)
		return (free(first_line), 0);
	startptr = first_line;
	count = count_commas(first_line) + 1;
	if (count < 2)
		return (free(first_line), 0);
	shape = calloc(count, sizeof(size_t));
	if (!shape)
		return (free(first_line), 0);
	for (size_t i = 0; i < count; i++)
	{
		if (parse_size(&startptr, shape + i) == -1)
			return (free(shape), free(first_line), 0);
	}
	free(first_line);
	*out = shape;
	return (count);
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

static int	parse_double(char	**p, double	*out)
{
	char	*endptr = NULL;
	double	tmp;

	tmp = strtod(*p, &endptr);
	if (!endptr || *p == endptr ||
		(*endptr != ',' && *endptr != '\n' && *endptr != '\0')) 
		return (-1);
	*out = tmp;
	*p = endptr + 1;
	if (*endptr == ',')
		return (1);
	if (*endptr == '\n' || *endptr == '\0')
		return (0);
	return (-1);
}

static int	read_weights(t_model *model, FILE *file)
{
	char	*line = NULL;
	char	*ptr = NULL;
	size_t	line_size = 0;
	double	weight = 0;

	for (size_t i = 0; i < model->layer_count; i++)
	{
		for (size_t j = 0; j < model->layers[i].l_neurons; j++)
		{
			if (getline(&line, &line_size, file) <= 0)
				return (free(line), -1);
			ptr = line;
			for (size_t k = 0; k < model->layers[i].l_inputs; k++)
			{
				if (parse_double(&ptr, &weight) != 1)
					return(free(line), -1);
				model->layers[i].w[j * model->layers[i].l_inputs + k] = weight;
			}
			if (parse_double(&ptr, &weight) != 0)
				return(free(line), -1);
			model->layers[i].b[j] = weight;
		}
	}
	return(free(line), 0);
}

t_model	*load_model(char *filename)
{
	t_model	*model = NULL;
	size_t	*model_shape = NULL;
	size_t	model_size = 0;

	FILE *file = fopen(filename, "r");
	if (!file)
		return (NULL);
	model_size = parse_shape(file, &model_shape);
	if (model_size < 2)
		return (fclose(file), NULL);
	model = create_model(model_shape, model_size);
	if (!model)
		return (free(model_shape), fclose(file), NULL);
	if (read_weights(model, file) == -1)
		return(free_model(model), free(model_shape), fclose(file), NULL);
	return (fclose(file), model);
}

double	predict_neuron(double *weights, double bias, double *input, size_t size)
{
	double	z = 0;

	for(size_t i = 0; i < size; i++)
		z += weights[i] * input[i];
	z += bias;
	return (1.0 / (1.0 + exp(-z)));
}

double	*forward(t_model *model, double *input)
{
	double	*current_input = input;
	double	*model_output = NULL;

	for (size_t i = 0; i < model->layer_count; i++)
	{
		model_output = calloc(model->layers[i].l_neurons, sizeof(double));
		if (!model_output)
		{
			if (current_input != input)
				free(current_input);
			return (NULL);
		}
		for (size_t j = 0; j < model->layers[i].l_neurons; j++)
		{
			model_output[j] = predict_neuron(
				&model->layers[i].w[j * model->layers[i].l_inputs],
				model->layers[i].b[j], current_input, model->layers[i].l_inputs);
		}
		if (current_input != input)
			free(current_input);
		current_input = model_output;
	}
	return (model_output);
}
