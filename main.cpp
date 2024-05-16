#include <iostream>
#include <string>
#include <Windows.h>
#include "main.h"


#define KB(n) ((1<<10) * n)

int main(int argc, char *argv[]) {
	std::string command = {};
	static const char base_str[] = "py main.py";

	for (int i = 1; i < argc; i++)
	{
		command += argv[i];
		command += ' ';
	}


	return system(command.c_str());
}
