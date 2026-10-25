#include "model.h"

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
