NAME    = libmodel.a

CC      = cc
CFLAGS  = -Wall -Wextra -Werror -g

SRC     = model.c model_io.c model_forward.c activation.c model_trainer.c
OBJ     = $(SRC:.c=.o)

all: $(NAME)

$(NAME): $(OBJ)
	ar rcs $@ $^

%.o: %.c model.h activation.h model_trainer.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ)

fclean: clean
	rm -rf $(NAME)

re: fclean all

.PHONY: all clean fclean re
