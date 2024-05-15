#include <stdio.h>
#include <Windows.h>
#include "main.h"


#define KB(n) ((1<<10) * n)

int main(int argc, char *argv[]) {
	size_t command_cp = KB(64);
	char *command = calloc(command_cp, sizeof(*command));
	if (!command)
	{
		printf("FATEL ERROR: NO MEMORY FOR %lld BYTES", command_cp * sizeof(*command));
		return 1;
	}
	static const char base_str[] = "py main.py";


	strcpy(command, base_str);


	size_t index = sizeof(base_str) - 1;

	for (int i = 1; i < argc; i++)
	{
		const size_t arg_ln = strlen(argv[i]);
		if (arg_ln + 1 >= command_cp - index)
		{
			command_cp <<= 2;
			void *c = command;
			command = realloc(command, command_cp * sizeof(*command));
			if (!command)
			{
				free(c);
				printf("FATEL ERROR WHILE PROCESSING ARG %d: NO MEMORY FOR %lld BYTES",
							 i, command_cp * sizeof(*command));
				return 1;
			}

			command[index++] = ' ';
			strcpy(command + index, argv[i]);

			index += arg_ln;
		}
	}

	return system(command);
}
