#ifndef IILS_STATISTICS_HPP
#define IILS_STATISTICS_HPP

#include <iosfwd>
#include <string>

namespace iils {

enum class InputType {
    GeneTree,
    MsaList,
};

struct StatisticsOptions {
    InputType input_type = InputType::GeneTree;
    std::string input_path;
    std::string constraint_tree_path;
    std::string mapping_path;
    int threads = 1;
    int chunk_size = 10000;
    int seed = 233;
    int gene_id_offset = 0;
};

// Writes branch-major, per-item statistics. No support interpretation, loss,
// threshold, filtering, or species-tree search is performed by this interface.
void write_statistics(
    const StatisticsOptions& options,
    std::ostream& branch_output,
    std::ostream* item_output = nullptr
);

// Versioned binary request/response protocol for the one-shot executor.
void serve_statistics(const StatisticsOptions& options, std::istream& input,
                      std::ostream& output);

}  // namespace iils

#endif
