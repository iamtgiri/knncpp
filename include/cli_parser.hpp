#pragma once
#include "cli_options.hpp"
#include <string>

inline CLIOptions parse_args(int argc, char *argv[])
{
    CLIOptions opt;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if ((arg == "--data" || arg == "-d") && i + 1 < argc)
        {
            opt.data = argv[++i];
        }
        else if ((arg == "--test_size" || arg == "-r") && i + 1 < argc)
        {
            opt.test_size = std::stof(argv[++i]);
        }
        else if ((arg == "--k_neighbors" || arg == "-k") && i + 1 < argc)
        {
            opt.k_neighbors = std::stoi(argv[++i]);
        }
        else if ((arg == "--mode" || arg == "-m") && i + 1 < argc)
        {
            opt.mode = argv[++i];
        }
        else if (arg == "--parallel" || arg == "-p")
        {
            opt.parallel = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            opt.help = true;
        }
        else
        {
            std::cerr << "[ERROR] Unknown option: " << arg << "\n";
            opt.help = true;
            return opt;
        }
    }

    return opt;
}
