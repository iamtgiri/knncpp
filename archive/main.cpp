#include "cxxopts.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    cxxopts::Options options("TestApp", "Testing cxxopts");
    options.add_options()("n,number", "Some number", cxxopts::value<int>()->default_value("42"))
    ("h,help", "Print usage")
    ("u,user", "User name", cxxopts::value<std::string>()->default_value("guest"));

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }
    if (result.count("user"))
    {
        std::cout << "User: " << result["user"].as<std::string>() << "\n";
        return 0;
    }

    std::cout << "Number: " << result["number"].as<int>() << std::endl;
    return 0;
}
