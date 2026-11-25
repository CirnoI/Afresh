#include "iils/statistics.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    iils::StatisticsOptions statistics;
    std::string output_path;
    std::string item_output_path;
    bool serve = false;
};

void print_help(std::ostream& out) {
    out
        << "Usage: iils-statistics --input-type TYPE -c TREE [-o FILE] [options] INPUT\n"
        << "\n"
        << "Export per-item quartet/site-pattern statistics on a fixed species-tree topology.\n"
        << "The interface does not infer a tree or calculate support, loss, thresholds, or filters.\n"
        << "\n"
        << "Required:\n"
        << "  -c, --constraint FILE    Fixed binary species-tree topology; annotations ignored\n"
        << "  --input-type TYPE        gene-tree or msa-list\n"
        << "\n"
        << "Output:\n"
        << "  -o, --output FILE        Branch statistics TSV (default: stdout)\n"
        << "  --item-output FILE       MSA locus metadata TSV\n"
        << "\n"
        << "Input options:\n"
        << "  -a, --mapping FILE       Gene/sequence name to species mapping\n"
        << "  --chunk N                MSA local-region size (default: 10000)\n"
        << "  -t, --thread N           ASTER worker threads (default: 1)\n"
        << "  --seed N                 Polytomy-resolution seed (default: 233)\n"
        << "  --gene-id-offset N       Add N to gene-tree output IDs (default: 0)\n"
        << "  --serve                  Binary one-shot worker protocol v1 on stdin/stdout\n"
        << "  -h, --help               Show this message\n";
}

int parse_positive_int(const std::string& text, const std::string& option) {
    char* end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value < 1) {
        throw std::runtime_error(option + " requires a positive integer");
    }
    return static_cast<int>(value);
}

int parse_nonnegative_int(const std::string& text, const std::string& option) {
    char* end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || value < 0) {
        throw std::runtime_error(option + " requires a nonnegative integer");
    }
    return static_cast<int>(value);
}

std::string take_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + option);
    }
    return argv[++index];
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    bool input_type_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(std::cout);
            std::exit(0);
        }
        if (arg == "--serve") {
            options.serve = true;
        }
        else if (arg == "-c" || arg == "--constraint") {
            options.statistics.constraint_tree_path = take_value(argc, argv, i, arg);
        }
        else if (arg == "-o" || arg == "--output") {
            options.output_path = take_value(argc, argv, i, arg);
        }
        else if (arg == "--item-output") {
            options.item_output_path = take_value(argc, argv, i, arg);
        }
        else if (arg == "-a" || arg == "--mapping") {
            options.statistics.mapping_path = take_value(argc, argv, i, arg);
        }
        else if (arg == "-t" || arg == "--thread" || arg == "--threads") {
            options.statistics.threads = parse_positive_int(
                take_value(argc, argv, i, arg), arg
            );
        }
        else if (arg == "--chunk") {
            options.statistics.chunk_size = parse_positive_int(
                take_value(argc, argv, i, arg), arg
            );
        }
        else if (arg == "--seed") {
            options.statistics.seed = parse_nonnegative_int(
                take_value(argc, argv, i, arg), arg
            );
        }
        else if (arg == "--gene-id-offset") {
            options.statistics.gene_id_offset = parse_nonnegative_int(
                take_value(argc, argv, i, arg), arg
            );
        }
        else if (arg == "--input-type") {
            std::string value = take_value(argc, argv, i, arg);
            if (value == "gene-tree") {
                options.statistics.input_type = iils::InputType::GeneTree;
            }
            else if (value == "msa-list") {
                options.statistics.input_type = iils::InputType::MsaList;
            }
            else {
                throw std::runtime_error("unsupported --input-type: " + value);
            }
            input_type_set = true;
        }
        else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("unknown option: " + arg);
        }
        else if (options.statistics.input_path.empty()) {
            options.statistics.input_path = arg;
        }
        else {
            throw std::runtime_error("multiple input files were provided");
        }
    }

    if (options.serve) return options;
    if (!input_type_set) {
        throw std::runtime_error("missing required --input-type");
    }
    if (options.statistics.constraint_tree_path.empty()) {
        throw std::runtime_error("missing required --constraint");
    }
    if (options.statistics.input_path.empty()) {
        throw std::runtime_error("missing input file");
    }
    if (
        options.statistics.input_type == iils::InputType::MsaList &&
        options.item_output_path.empty()
    ) {
        throw std::runtime_error("msa-list input requires --item-output");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CliOptions options = parse_args(argc, argv);
        if (options.serve) {
            iils::serve_statistics(options.statistics, std::cin, std::cout);
            return 0;
        }
        std::ofstream branch_file;
        std::ostream* branch_output = &std::cout;
        if (!options.output_path.empty()) {
            branch_file.open(options.output_path);
            if (!branch_file) {
                throw std::runtime_error(
                    "unable to open branch output: " + options.output_path
                );
            }
            branch_output = &branch_file;
        }

        std::ofstream item_file;
        std::ostream* item_output = nullptr;
        if (!options.item_output_path.empty()) {
            item_file.open(options.item_output_path);
            if (!item_file) {
                throw std::runtime_error(
                    "unable to open item output: " + options.item_output_path
                );
            }
            item_output = &item_file;
        }

        iils::write_statistics(options.statistics, *branch_output, item_output);
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
