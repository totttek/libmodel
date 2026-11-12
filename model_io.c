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

static int	is_correct_version(FILE *file)
{
	char	*line = NULL;
	char	*endptr = NULL;
	size_t	line_size = 0;
	long	version = 0;
	size_t	len = 0;

	if (getline(&line, &line_size, file) <= 0)
		return (free(line), 0);
	len = strlen(MODEL_FORMAT_NAME);
	if (strncmp(MODEL_FORMAT_NAME, line, len))
		return(free(line), 0);
	version = strtol(line + len, &endptr, 10);
	if (line + len == endptr || *endptr != '\n'|| version != MODEL_FORMAT_VERSION)
		return (free(line), 0);
	free(line);
	return (1);
}

static int	parse_activation(char	**p, t_activation *out)
{
	char	*str = NULL;
	size_t	len = 0;
	t_activation	tmp;

	str = strtok(*p, ",\n");
	if (!str)
		return (-1);
	len = strlen(str) + 1;
	*p += len;
	if (activation_from_name(str, &tmp) != 0)
		return(-1);
	*out = tmp;
	return (0);
}

static int	read_activations(t_model *model, FILE *file)
{
	char	*line = NULL;
	char	*startptr = NULL;
	size_t	line_size = 0;

	if (getline(&line, &line_size, file) <= 0)
		return (free(line), -1);
	if (line[strlen(line) - 1] != '\n')
		return(free(line), -1);
	startptr = line;
	for (size_t i = 0; i < model->layer_count; i++)
	{
		if (parse_activation(&startptr, &(model->layers[i].act)) != 0)
			return(free(line), -1);
	}
	if (*startptr != '\0')
		return(free(line), -1);
	return(free(line), 0);
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

static size_t	read_shape(FILE *file, size_t **out)
{
	char	*line = NULL;
	char	*startptr = NULL;
	size_t	*shape = NULL;
	size_t	line_size = 0;
	size_t	count = 0;

	if (getline(&line, &line_size, file) <= 0)
		return (free(line), 0);
	startptr = line;
	count = count_commas(line) + 1;
	if (count < 2)
		return (free(line), 0);
	shape = calloc(count, sizeof(size_t));
	if (!shape)
		return (free(line), 0);
	for (size_t i = 0; i < count; i++)
	{
		if (parse_size(&startptr, shape + i) == -1)
			return (free(shape), free(line), 0);
	}
	free(line);
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

t_model	*model_load(char *filename)
{
	t_model	*model = NULL;
	size_t	*model_shape = NULL;
	size_t	model_size = 0;

	FILE *file = fopen(filename, "r");
	if (!file)
		return (NULL);
	if (!is_correct_version(file))
		return (fclose(file), NULL);
	model_size = read_shape(file, &model_shape);
	if (model_size < 2)
		return (fclose(file), NULL);
	model = model_create(model_shape, model_size);
	if (!model)
		return (free(model_shape), fclose(file), NULL);
	if (read_activations(model, file) != 0)
		return(model_free(model), free(model_shape), fclose(file), NULL);
	if (read_weights(model, file) != 0)
		return(model_free(model), free(model_shape), fclose(file), NULL);
	return (fclose(file), free(model_shape), model);
}

int	model_save(t_model *model, char *filename)
{
	const t_act_funs	*tmp = NULL;

	FILE *file = fopen(filename, "w");
	if (!file)
		return(-1);
	fprintf(file, "%s%d\n", MODEL_FORMAT_NAME, MODEL_FORMAT_VERSION);
	fprintf(file, "%zu", model->layers[0].l_inputs);
	for (size_t i = 0; i < model->layer_count; i++)
	{
		fprintf(file, ",%zu", model->layers[i].l_neurons);
	}
	fprintf(file, "\n");
	for (size_t i = 0; i < model->layer_count; i++)
	{
		tmp = activation_get(model->layers[i].act);
		if (!tmp)
			return(fclose(file), -1);
		fprintf(file, "%s", tmp->name);
		if (i == model->layer_count - 1)
			fprintf(file, "\n");
		else
			fprintf(file, ",");
	}
	for (size_t i = 0; i < model->layer_count; i++)
	{
		for (size_t j = 0; j < model->layers[i].l_neurons; j++)
		{
			for (size_t k = 0; k < model->layers[i].l_inputs; k++)
			{
				fprintf(file, "%.17g,", model->layers[i].w[j * model->layers[i].l_inputs + k]);
			}
			fprintf(file, "%.17g\n", model->layers[i].b[j]);
		}
	}
	if (fclose(file))
		return (-1);
	return (0);
}
