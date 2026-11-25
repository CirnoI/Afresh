#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "iils/statistics.hpp"

using namespace std;

#define DRIVER_VERSION "0"

typedef long long score_t;
typedef double length_t;

double from_string(const string s) {
	return stold(s);
}

#include "argparser.hpp"
#include "genetreewithbinaryweight.hpp"
#include "treeutils.hpp"
#include "algorithms.hpp"

#undef SUPPORT
#undef G_SUPPORT
#undef LOCAL_BOOTSTRAP
#undef CUSTOMIZED_ANNOTATION
#undef SOME_FORM_OF_SUPPORT
#undef OBJECTIVE_VERSION
#undef THREAD_POOL

namespace msa_backend {
typedef double score_t;
typedef int count_t;

#define BLOCK_BOOTSTRAP
#ifndef CUSTOMIZED_ANNOTATION_TERMINAL_LENGTH
#define CUSTOMIZED_ANNOTATION_LENGTH
#endif

#include "sequence.hpp"
#include "algorithms.hpp"
#include "sequtils.hpp"
}

bool isNewickDelimiter(char c) {
	return c == ':' || c == ',' || c == ')' || c == ';';
}

size_t skipSpaces(const string& text, size_t pos) {
	while (pos < text.size() && isspace((unsigned char) text[pos])) pos++;
	return pos;
}

size_t skipQuotedToken(const string& text, size_t pos) {
	char quote = text[pos++];
	while (pos < text.size()) {
		if (text[pos] == quote) {
			pos++;
			break;
		}
		pos++;
	}
	return pos;
}

size_t skipBracketToken(const string& text, size_t pos) {
	int depth = 0;
	while (pos < text.size()) {
		if (text[pos] == '\'' || text[pos] == '"') {
			pos = skipQuotedToken(text, pos);
			continue;
		}
		if (text[pos] == '[') depth++;
		if (text[pos] == ']') {
			depth--;
			pos++;
			if (depth <= 0) break;
			continue;
		}
		pos++;
	}
	return pos;
}

size_t skipInternalNodeLabel(const string& text, size_t pos) {
	pos = skipSpaces(text, pos);
	bool skipped = true;
	while (skipped && pos < text.size() && !isNewickDelimiter(text[pos])) {
		skipped = false;
		if (text[pos] == '\'' || text[pos] == '"') {
			pos = skipQuotedToken(text, pos);
			skipped = true;
		}
		else if (text[pos] == '[') {
			pos = skipBracketToken(text, pos);
			skipped = true;
		}
		else {
			while (pos < text.size() && !isNewickDelimiter(text[pos])) pos++;
			skipped = true;
		}
		pos = skipSpaces(text, pos);
	}
	return pos;
}

string stripInternalNodeLabels(const string& text) {
	string res;
	bool singleQuote = false, doubleQuote = false;
	int bracketDepth = 0;
	for (size_t pos = 0; pos < text.size();) {
		char c = text[pos];
		res.push_back(c);
		pos++;
		if (c == '\'' && !doubleQuote) singleQuote = !singleQuote;
		else if (c == '"' && !singleQuote) doubleQuote = !doubleQuote;
		else if (!singleQuote && !doubleQuote) {
			if (c == '[') bracketDepth++;
			else if (c == ']' && bracketDepth > 0) bracketDepth--;
			else if (c == ')' && bracketDepth == 0) {
				pos = skipInternalNodeLabel(text, pos);
			}
		}
	}
	return res;
}

string readWholeFile(const string& path, const string& description) {
	ifstream fin(path);
	if (!fin.is_open()) throw runtime_error("Unable to open " + description + " file: " + path);
	string line, text;
	while (getline(fin, line)) text += line;
	if (text.empty()) throw runtime_error(description + " file is empty: " + path);
	return text;
}

string trimCopy(const string& s) {
	size_t i = 0, j = s.size();
	while (i < j && isspace((unsigned char) s[i])) i++;
	while (j > i && isspace((unsigned char) s[j - 1])) j--;
	return s.substr(i, j - i);
}

string unquoteToken(string s) {
	s = trimCopy(s);
	if (s.size() >= 2 && ((s.front() == '\'' && s.back() == '\'') || (s.front() == '"' && s.back() == '"'))) {
		s = s.substr(1, s.size() - 2);
	}
	return s;
}

string canonicalSplitKey(vector<int> clade, int nTaxa) {
	sort(clade.begin(), clade.end());
	clade.erase(unique(clade.begin(), clade.end()), clade.end());
	vector<int> other;
	size_t p = 0;
	for (int i = 0; i < nTaxa; i++) {
		if (p < clade.size() && clade[p] == i) p++;
		else other.push_back(i);
	}
	if (clade.size() > other.size() || (clade.size() == other.size() && other < clade)) clade.swap(other);
	string key;
	for (int x: clade) {
		if (!key.empty()) key += ",";
		key += to_string(x);
	}
	return key;
}

void writeGeneStatisticsHeader(ostream& out) {
	out << setprecision(12)
		<< "gene_id\tsplit_key\tdiscordance_mass\n";
}

void writeMsaStatisticsHeader(ostream& out) {
	out << setprecision(12)
		<< "locus_id\talignment_path\tspecies_branch_id\tsplit_key"
		<< "\tsequence_score1\tsequence_score2\tsequence_score3"
		<< "\tsequence_score_sum\tq_locus\tn_sites_used\tn_taxa_present\n";
}

void writeMsaItemHeader(ostream& out) {
	out << "locus_id\talignment_path\tn_taxa_present\tn_sites_raw"
		<< "\tn_sites_used\tn_internal_gene_blocks\tn_species_branches\n";
}

struct GeneTreeReader {
	TripartitionInitializer& tripInit;
	vector<string>& names;
	unordered_map<string, int>& name2id;
	unordered_map<string, string> leafnameMapping;
	unordered_set<int> speciesTaxa;
	string text;
	long long pos = 0;
	int nGeneTrees = 0;
	int part = 0;
	std::mt19937_64 polytomyRandom;

	GeneTreeReader(TripartitionInitializer& tripInit, vector<string>& names, unordered_map<string, int>& name2id):
		tripInit(tripInit), names(names), name2id(name2id) {}

	void readMapping(const string& mappingPath) {
		if (mappingPath.empty()) return;
		ifstream fin(mappingPath);
		if (!fin.is_open()) {
			throw runtime_error("Unable to open mapping file: " + mappingPath);
		}
		string geneName, speciesName;
		while (fin >> geneName >> speciesName) {
			leafnameMapping[geneName] = speciesName;
		}
	}

	string mappedName(const string& name) const {
		auto it = leafnameMapping.find(name);
		return it == leafnameMapping.end() ? name : it->second;
	}

	void registerSpeciesName(const string& name) {
		string realName = name;
		if (name2id.count(realName) == 0) {
			int id = names.size();
			name2id[realName] = id;
			names.push_back(realName);
			for (size_t i = 0; i < tripInit.leafParent.size(); i++) {
				tripInit.leafParent[i].emplace_back();
			}
		}
		speciesTaxa.insert(name2id[realName]);
	}

	void registerSpeciesTreeTaxa(const string& speciesTreeText) {
		Tree speciesTree(stripInternalNodeLabels(speciesTreeText));
		for (const Tree::Node& node: speciesTree.nodes) {
			if (node.isleaf) registerSpeciesName(node.label);
		}
	}

	int mapLeaf(long long begin, long long end) {
		string s;
		for (long long i = begin; i < end && text[i] != ':'; i++) {
			if (text[i] != '\"' && text[i] != '\'') s += text[i];
		}
		s = mappedName(s);
		if (name2id.count(s) == 0 || speciesTaxa.count(name2id[s]) == 0) {
			throw runtime_error("Gene tree leaf '" + s + "' is not present in the species tree");
		}
		return name2id[s];
	}

	double length(long long begin, long long end) {
		long long i = begin;
		while (i < end && text[i] != ':') i++;
		if (i == end) return 0;
		return from_string(text.substr(i + 1, end - i - 1));
	}

	void parseSubtree(int parent = -1, bool isLeft = true) {
		int cur = tripInit.nodes[part].size();
		tripInit.nodes[part].emplace_back();
		tripInit.nodes[part][cur].up = parent;
		if (parent != -1 && isLeft) tripInit.nodes[part][parent].small = cur;
		if (parent != -1 && !isLeft) tripInit.nodes[part][parent].large = cur;

		if (text[pos] == '(') {
			pos++;
			parseSubtree(cur, true);
			pos++;
			parseSubtree(cur, false);
			vector<int> lst;
			lst.push_back(cur);
			tripInit.nodes[part][cur].isGhostBranch = true;
			tripInit.nodes[part][cur].length = 0;
			while (text[pos] != ')') {
				int left = lst[polytomyRandom() % lst.size()];
				int up = tripInit.nodes[part].size();
				tripInit.nodes[part].emplace_back();
				lst.push_back(up);
				if (cur == left) cur = up;
				tripInit.nodes[part][up].isGhostBranch = true;
				tripInit.nodes[part][up].length = 0;
				int g = tripInit.nodes[part][left].up;
				if (g != -1) {
					if (tripInit.nodes[part][g].small == left) tripInit.nodes[part][g].small = up;
					else tripInit.nodes[part][g].large = up;
				}
				tripInit.nodes[part][up].up = g;
				tripInit.nodes[part][left].up = up;
				tripInit.nodes[part][up].small = left;
				pos++;
				parseSubtree(up, false);
			}
			long long i = ++pos;
			while (text[pos] != ')' && text[pos] != ',' && text[pos] != ';') pos++;
			tripInit.nodes[part][cur].isGhostBranch = false;
			tripInit.nodes[part][cur].length = length(i, pos);
		}
		else {
			long long i = pos;
			while (text[pos] != ')' && text[pos] != ',') pos++;
			int taxon = mapLeaf(i, pos);
			tripInit.leafParent[part][taxon].push_back(cur);
			tripInit.nodes[part][cur].isGhostBranch = false;
			tripInit.nodes[part][cur].length = length(i, pos);
		}
	}

	void readGeneTrees(const string& inputPath, int seed, int geneIdOffset) {
		ifstream fin(inputPath);
		if (!fin.is_open()) {
			throw runtime_error("Unable to open input gene tree file: " + inputPath);
		}
		string line;
		while (getline(fin, line)) text += line;
		text = stripInternalNodeLabels(text);
		while (pos < (long long) text.size()) {
			while (pos < (long long) text.size() && text[pos] != '(') pos++;
			if (pos < (long long) text.size()) {
				part = nGeneTrees % tripInit.nodes.size();
				// Each global gene ID owns its stream, independently of process chunks.
				std::seed_seq identity{static_cast<unsigned>(seed),
					static_cast<unsigned>(geneIdOffset + nGeneTrees)};
				polytomyRandom.seed(identity);
				parseSubtree();
				nGeneTrees++;
			}
		}
	}

	void readGeneTreeRecords(const vector<pair<int,string>>& records, int seed) {
		for (const auto& record: records) {
			if (record.first < 0) throw runtime_error("gene ID must be nonnegative");
			text = record.second;
			pos = 0;
			text = stripInternalNodeLabels(text);
			while (pos < (long long) text.size() && text[pos] != '(') pos++;
			if (pos >= (long long) text.size()) throw runtime_error("gene tree record is empty");
			part = nGeneTrees % tripInit.nodes.size();
			std::seed_seq identity{static_cast<unsigned>(seed), static_cast<unsigned>(record.first)};
			polytomyRandom.seed(identity);
			parseSubtree();
			nGeneTrees++;
		}
	}
};

vector<int> partitionRoots(const Quadrupartition::Partition& part) {
	vector<int> roots;
	for (int i = 0; i < (int) part.nodes.size(); i++) {
		if (part.nodes[i].up == -1) roots.push_back(i);
	}
	return roots;
}

vector<int> collectSubtreeLeaves(ConstrainedOptimizationAlgorithm& alg, int v) {
	if (alg.nodes[v].leafId != -1) return {alg.nodes[v].leafId};
	tuple<int, int, score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];
	vector<int> left = collectSubtreeLeaves(alg, get<0>(c));
	vector<int> right = collectSubtreeLeaves(alg, get<1>(c));
	left.insert(left.end(), right.begin(), right.end());
	sort(left.begin(), left.end());
	left.erase(unique(left.begin(), left.end()), left.end());
	return left;
}

vector<int> collectMsaSubtreeLeaves(msa_backend::ConstrainedOptimizationAlgorithm& alg, int v) {
	if (alg.nodes[v].leafId != -1) return {alg.nodes[v].leafId};
	tuple<int, int, msa_backend::score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];
	vector<int> left = collectMsaSubtreeLeaves(alg, get<0>(c));
	vector<int> right = collectMsaSubtreeLeaves(alg, get<1>(c));
	left.insert(left.end(), right.begin(), right.end());
	sort(left.begin(), left.end());
	left.erase(unique(left.begin(), left.end()), left.end());
	return left;
}

void writeGeneStatistics(
	Quadrupartition& quad,
	int nGeneTrees,
	ostream& out,
	const string& splitKey,
	int geneIdOffset)
{
	int nParts = quad.parts.size();
	for (int p = 0; p < nParts; p++) {
		vector<int> roots = partitionRoots(quad.parts[p]);
		for (int local = 0; local < (int) roots.size(); local++) {
			int geneId = p + local * nParts;
			if (geneId >= nGeneTrees) continue;
			int root = roots[local];
			const Quadrupartition::Partition::Node& rootNode = quad.parts[p].nodes[root];
			double qCapacity = (double) ((long double) rootNode.a * rootNode.b * rootNode.c * rootNode.d);
			if (qCapacity <= 0.0) continue;
			double q1 = quad.parts[p].score2[root];
			double q2 = quad.parts[p].score3[root];
			out << geneId + geneIdOffset << "\t" << splitKey << "\t"
				<< (q1 + q2) / qCapacity << "\n";
		}
	}
}

template<typename Visitor>
void visitBranchForIILS(
	ConstrainedOptimizationAlgorithm& alg,
	Quadrupartition& quad,
	int branchSubtree,
	Visitor& visitor)
{
	vector<int> clade = collectSubtreeLeaves(alg, branchSubtree);
	visitor(quad, clade);
}

template<typename Visitor>
void visitSubtreeForIILS(
	ConstrainedOptimizationAlgorithm& alg,
	Quadrupartition& quad,
	int v,
	int u,
	Visitor& visitor)
{
	if (alg.nodes[v].leafId != -1) return;
	tuple<int, int, score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];

	alg.switchSubtree(quad, u, 1, 0);
	alg.switchSubtree(quad, get<1>(c), 2, 1);
	visitSubtreeForIILS(alg, quad, get<0>(c), get<1>(c), visitor);

	alg.switchSubtree(quad, get<0>(c), 2, 1);
	alg.switchSubtree(quad, get<1>(c), 1, 2);
	visitSubtreeForIILS(alg, quad, get<1>(c), get<0>(c), visitor);

	alg.switchSubtree(quad, u, 0, 1);
	alg.switchSubtree(quad, get<0>(c), 1, 3);
	visitBranchForIILS(alg, quad, v, visitor);
	alg.switchSubtree(quad, get<0>(c), 3, 2);
}

template<typename Visitor>
void visitBranchesForIILS(ConstrainedOptimizationAlgorithm& alg, Visitor& visitor) {
	Quadrupartition quad(alg.tripInit);
	quad.update(0, 0);
	for (size_t i = 1; i < alg.names.size(); i++) quad.update(2, i);
	int v = alg.hash[-alg.taxonHash[0]];
	if (alg.nodes[v].leafId != -1) return;
	tuple<int, int, score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];

	alg.switchSubtree(quad, get<1>(c), 2, 1);
	visitSubtreeForIILS(alg, quad, get<0>(c), get<1>(c), visitor);

	alg.switchSubtree(quad, get<1>(c), 1, 2);
	alg.switchSubtree(quad, get<0>(c), 2, 1);
	visitSubtreeForIILS(alg, quad, get<1>(c), get<0>(c), visitor);
}

string toFormatNA(const string& seq) {
	string res;
	for (char c: seq) {
		switch (c) {
			case 'A': case 'a': res += 'A'; break;
			case 'C': case 'c': res += 'C'; break;
			case 'G': case 'g': res += 'G'; break;
			case 'T': case 't': case 'U': case 'u': res += 'T'; break;
			default: res += '-';
		}
	}
	return res;
}

vector<string> readLocusList(const string& path) {
	ifstream fin(path);
	if (!fin.is_open()) throw runtime_error("Unable to open MSA locus list: " + path);
	vector<string> paths;
	string line;
	while (getline(fin, line)) {
		line = trimCopy(line);
		if (line.empty() || line[0] == '#') continue;
		paths.push_back(line);
	}
	if (paths.empty()) throw runtime_error("MSA locus list is empty: " + path);
	return paths;
}

struct FastaAlignment {
	vector<pair<string, string> > records;
	long long length = 0;
};

FastaAlignment parseFastaAlignment(istream& fin, const string& path) {
	FastaAlignment aln;
	string line, name, seq;
	auto flushRecord = [&]() {
		if (name.empty()) return;
		string formatted = toFormatNA(seq);
		if (aln.records.empty()) aln.length = formatted.size();
		else if ((long long) formatted.size() != aln.length) {
			throw runtime_error("Sequence length is not consistent in FASTA alignment: " + path);
		}
		aln.records.emplace_back(name, formatted);
	};
	while (getline(fin, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		if (line[0] == '>') {
			flushRecord();
			name.clear();
			seq.clear();
			size_t i = 1;
			while (i < line.size() && isspace((unsigned char) line[i])) i++;
			while (i < line.size() && !isspace((unsigned char) line[i])) name.push_back(line[i++]);
			if (name.empty()) throw runtime_error("Unable to parse FASTA record name in: " + path);
		}
		else {
			for (char c: line) {
				if (!isspace((unsigned char) c)) seq.push_back(c);
			}
		}
	}
	flushRecord();
	if (aln.records.empty()) throw runtime_error("FASTA alignment is empty: " + path);
	return aln;
}

FastaAlignment readFastaAlignment(const string& path) {
	ifstream fin(path);
	if (!fin.is_open()) throw runtime_error("Unable to open FASTA alignment: " + path);
	return parseFastaAlignment(fin, path);
}

struct MsaLocusRecord {
	int locusId = 0;
	string alignmentPath;
	int firstGeneBlock = 0;
	int lastGeneBlock = 0;
	long long nSitesRaw = 0;
	long long nSitesUsed = 0;
	int nTaxaPresent = 0;
	vector<int> presentTaxa;
};

struct MsaSiteReader {
	msa_backend::TripartitionInitializer& tripInit;
	vector<string>& names;
	unordered_map<string, int>& name2id;
	unordered_map<string, string> leafnameMapping;
	unordered_set<int> speciesTaxa;
	vector<MsaLocusRecord> loci;

	MsaSiteReader(msa_backend::TripartitionInitializer& tripInit, vector<string>& names, unordered_map<string, int>& name2id):
		tripInit(tripInit), names(names), name2id(name2id) {}

	void readMapping(const string& mappingPath) {
		if (mappingPath.empty()) return;
		ifstream fin(mappingPath);
		if (!fin.is_open()) throw runtime_error("Unable to open mapping file: " + mappingPath);
		string geneName, speciesName;
		while (fin >> geneName >> speciesName) {
			leafnameMapping[geneName] = speciesName;
		}
	}

	string mappedName(const string& name) const {
		auto it = leafnameMapping.find(name);
		return it == leafnameMapping.end() ? name : it->second;
	}

	void registerSpeciesName(const string& name) {
		if (name2id.count(name) == 0) {
			int id = names.size();
			name2id[name] = id;
			names.push_back(name);
		}
		speciesTaxa.insert(name2id[name]);
	}

	void registerSpeciesTreeTaxa(const string& speciesTreeText) {
		Tree speciesTree(stripInternalNodeLabels(speciesTreeText));
		for (const Tree::Node& node: speciesTree.nodes) {
			if (node.isleaf) registerSpeciesName(node.label);
		}
	}

	void buildGeneSeq(msa_backend::TripartitionInitializer::Gene::Initializer& gene,
		const vector<int>& ind2species,
		size_t pos,
		size_t offset,
		const array<double, 4>& eqfreq)
	{
		for (int iInd = 0; iInd < (int) ind2species.size(); iInd++) {
			size_t pSeq = pos + iInd * offset;
			gene.species2ind[ind2species[iInd]].push_back(iInd);
			gene.ind2seq[iInd] = pSeq;
		}
		for (int j = 0; j < 4; j++) {
			gene.pi[j] = eqfreq[j];
		}
	}

	void formatGene(const vector<int>& ind2species, size_t pos, size_t nSite, size_t offset, const array<double, 4>& eqfreq) {
		int nInd = ind2species.size(), nSpecies = names.size(), nKernal = nSite, nRep = 0;
		msa_backend::TripartitionInitializer::Gene::Initializer gene(nInd, nSpecies, nSite, nKernal, nRep);
		buildGeneSeq(gene, ind2species, pos, offset, eqfreq);
		tripInit.genes.emplace_back(gene);
	}

	void readLocus(const string& path, int locusId, int chunkSize, const FastaAlignment* supplied = nullptr) {
		FastaAlignment aln = supplied ? *supplied : readFastaAlignment(path);
		long long nSites = aln.length;
		if (nSites <= 0) throw runtime_error("FASTA alignment has zero length: " + path);
		long long nChunk = (nSites + chunkSize - 1) / chunkSize;
		vector<vector<long long> > sites(nChunk);
		vector<array<double, 4> > eqfreq;
		long long offset = 0;
		vector<int> ind2species;
		vector<string> seqs;
		unordered_set<int> present;
		vector<array<unsigned short, 4> > freq(nSites);

		for (const auto& rec: aln.records) {
			string sname = mappedName(rec.first);
			auto it = name2id.find(sname);
			if (it == name2id.end() || speciesTaxa.count(it->second) == 0) {
				throw runtime_error("MSA sequence '" + rec.first + "' maps to species '" + sname + "', which is not present in the species tree");
			}
			int species = it->second;
			bool hasUsableBase = false;
			for (long long i = 0; i < nSites; i++) {
				switch (rec.second[i]) {
					case 'A': freq[i][0]++; hasUsableBase = true; break;
					case 'C': freq[i][1]++; hasUsableBase = true; break;
					case 'G': freq[i][2]++; hasUsableBase = true; break;
					case 'T': freq[i][3]++; hasUsableBase = true; break;
				}
			}
			if (!hasUsableBase) continue;
			ind2species.push_back(species);
			present.insert(species);
			seqs.push_back(rec.second);
		}


		for (long long i = 0; i < nChunk; i++) {
			long long s = i * nSites / nChunk, t = (i + 1) * nSites / nChunk;
			long long sumFreq[4] = {};
			for (long long j = s; j < t; j++) {
				for (int k = 0; k < 4; k++) sumFreq[k] += freq[j][k];
				if (freq[j][0] + freq[j][2] >= 2 && freq[j][1] + freq[j][3] >= 2) sites[i].push_back(j);
			}
			offset += sites[i].size();
			double total = sumFreq[0] + sumFreq[1] + sumFreq[2] + sumFreq[3];
			if (total > 0) eqfreq.push_back({ sumFreq[0] / total, sumFreq[1] / total, sumFreq[2] / total, sumFreq[3] / total });
			else eqfreq.push_back({ 0.25, 0.25, 0.25, 0.25 });
		}

		MsaLocusRecord locus;
		locus.locusId = locusId;
		locus.alignmentPath = path;
		locus.firstGeneBlock = tripInit.genes.size();
		locus.nSitesRaw = nSites;
		locus.nSitesUsed = offset;
		locus.nTaxaPresent = present.size();
		locus.presentTaxa.assign(present.begin(), present.end());
		sort(locus.presentTaxa.begin(), locus.presentTaxa.end());

		size_t pos = tripInit.seq.len();
		for (const string& seq: seqs) {
			for (long long i = 0; i < nChunk; i++) {
				for (long long j: sites[i]) tripInit.seq.append(seq[j]);
			}
		}
		for (int i = 0; i < (int) nChunk; i++) {
			formatGene(ind2species, pos, sites[i].size(), offset, eqfreq[i]);
			pos += sites[i].size();
		}
		locus.lastGeneBlock = tripInit.genes.size();
		loci.push_back(locus);
	}

	void readLoci(const vector<string>& paths, int chunkSize) {
		for (int i = 0; i < (int) paths.size(); i++) {
			readLocus(paths[i], i, chunkSize);
		}
		tripInit.nSpecies = names.size();
	}
};

array<double, 3> scoreMsaLocusForCurrentColoring(
	const MsaLocusRecord& locus,
	msa_backend::Quadrupartition& quad)
{
	array<double, 3> result = {};
	for (int g = locus.firstGeneBlock; g < locus.lastGeneBlock; g++) {
		array<msa_backend::score_t, 3> temp = quad.genes[g].scoreCnt();
		for (int i = 0; i < 3; i++) result[i] += temp[i];
	}
	return result;
}

long long msaQuartetCapacityForCurrentColoring(
	const MsaLocusRecord& locus,
	const msa_backend::Quadrupartition& quad)
{
	if (quad.color.empty()) return 0;
	array<long long, 4> cnt = {0, 0, 0, 0};
	const vector<char>& color = quad.color[0];
	for (int taxon: locus.presentTaxa) {
		if (taxon < 0 || taxon >= (int) color.size()) continue;
		int c = color[taxon];
		if (0 <= c && c < 4) cnt[c]++;
	}
	return cnt[0] * cnt[1] * cnt[2] * cnt[3];
}

void writeMsaBranchStatistics(
	msa_backend::Quadrupartition& quad,
	const vector<MsaLocusRecord>& loci,
	ostream& out,
	int speciesBranchId,
	const string& splitKey)
{
	for (const MsaLocusRecord& locus: loci) {
		if (locus.nTaxaPresent < 4 || locus.firstGeneBlock == locus.lastGeneBlock) {
			continue;
		}
		array<double, 3> score = scoreMsaLocusForCurrentColoring(locus, quad);
		double scoreSum = score[0] + score[1] + score[2];
		long long qLocus = msaQuartetCapacityForCurrentColoring(locus, quad);
		out << locus.locusId
			<< "\t" << locus.alignmentPath
			<< "\t" << speciesBranchId
			<< "\t" << splitKey
			<< "\t" << score[0]
			<< "\t" << score[1]
			<< "\t" << score[2]
			<< "\t" << scoreSum
			<< "\t" << qLocus
			<< "\t" << locus.nSitesUsed
			<< "\t" << locus.nTaxaPresent
			<< "\n";
	}
}

template<typename Visitor>
void visitMsaBranchForStatistics(
	msa_backend::ConstrainedOptimizationAlgorithm& alg,
	msa_backend::Quadrupartition& quad,
	int branchSubtree,
	Visitor& visitor)
{
	vector<int> clade = collectMsaSubtreeLeaves(alg, branchSubtree);
	visitor(quad, clade);
}

template<typename Visitor>
void visitMsaSubtreeForStatistics(
	msa_backend::ConstrainedOptimizationAlgorithm& alg,
	msa_backend::Quadrupartition& quad,
	int v,
	int u,
	Visitor& visitor)
{
	if (alg.nodes[v].leafId != -1) return;
	tuple<int, int, msa_backend::score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];

	alg.switchSubtree(quad, u, 1, 0);
	alg.switchSubtree(quad, get<1>(c), 2, 1);
	visitMsaSubtreeForStatistics(alg, quad, get<0>(c), get<1>(c), visitor);

	alg.switchSubtree(quad, get<0>(c), 2, 1);
	alg.switchSubtree(quad, get<1>(c), 1, 2);
	visitMsaSubtreeForStatistics(alg, quad, get<1>(c), get<0>(c), visitor);

	alg.switchSubtree(quad, u, 0, 1);
	alg.switchSubtree(quad, get<0>(c), 1, 3);
	visitMsaBranchForStatistics(alg, quad, v, visitor);
	alg.switchSubtree(quad, get<0>(c), 3, 2);
}

template<typename Visitor>
void visitMsaBranchesForStatistics(
	msa_backend::ConstrainedOptimizationAlgorithm& alg,
	Visitor& visitor)
{
	msa_backend::Quadrupartition quad(alg.tripInit);
	quad.update(0, 0);
	for (size_t i = 1; i < alg.names.size(); i++) quad.update(2, i);
	int v = alg.hash[-alg.taxonHash[0]];
	if (alg.nodes[v].leafId != -1) return;
	tuple<int, int, msa_backend::score_t> c = alg.nodes[v].children[alg.nodes[v].bestChild];

	alg.switchSubtree(quad, get<1>(c), 2, 1);
	visitMsaSubtreeForStatistics(alg, quad, get<0>(c), get<1>(c), visitor);

	alg.switchSubtree(quad, get<1>(c), 1, 2);
	alg.switchSubtree(quad, get<0>(c), 2, 1);
	visitMsaSubtreeForStatistics(alg, quad, get<1>(c), get<0>(c), visitor);
}

void writeMsaItems(
	const vector<MsaLocusRecord>& loci,
	int nSpeciesBranches,
	ostream& out)
{
	writeMsaItemHeader(out);
	for (const MsaLocusRecord& locus: loci) {
		out << locus.locusId
			<< "\t" << locus.alignmentPath
			<< "\t" << locus.nTaxaPresent
			<< "\t" << locus.nSitesRaw
			<< "\t" << locus.nSitesUsed
			<< "\t" << (locus.lastGeneBlock - locus.firstGeneBlock)
			<< "\t" << nSpeciesBranches
			<< "\n";
	}
}

void runMsaStatistics(
	const iils::StatisticsOptions& options,
	const string& speciesTreeText,
	ostream& branchOutput,
	ostream& itemOutput)
{
	vector<string> locusPaths = readLocusList(options.input_path);
	msa_backend::TripartitionInitializer tripInit;
	tripInit.nThreads = options.threads;
	vector<string> names;
	unordered_map<string, int> name2id;

	MsaSiteReader reader(tripInit, names, name2id);
	reader.readMapping(options.mapping_path);
	reader.registerSpeciesTreeTaxa(speciesTreeText);
	reader.readLoci(locusPaths, options.chunk_size);
	if (reader.loci.empty()) throw runtime_error("No loci were read from MSA input");

	msa_backend::ThreadPool threadPool;
	threadPool.initialize(options.threads);
	msa_backend::ConstrainedOptimizationAlgorithm alg(
		names.size(), tripInit, names, threadPool, -1
	);
	alg.addGuideTree(speciesTreeText, name2id, -1);
	alg.computeOptimalTree();

	writeMsaStatisticsHeader(branchOutput);
	struct Visitor {
		const vector<MsaLocusRecord>& loci;
		int nTaxa;
		ostream& out;
		int nextSpeciesBranchId;

		void operator()(
			msa_backend::Quadrupartition& quad,
			const vector<int>& clade)
		{
			string splitKey = canonicalSplitKey(clade, nTaxa);
			writeMsaBranchStatistics(
				quad, loci, out, nextSpeciesBranchId++, splitKey
			);
		}
	} visitor = {reader.loci, (int) names.size(), branchOutput, 0};
	visitMsaBranchesForStatistics(alg, visitor);
	writeMsaItems(reader.loci, visitor.nextSpeciesBranchId, itemOutput);
}

void runGeneTreeStatistics(
	const iils::StatisticsOptions& options,
	const string& speciesTreeText,
	ostream& branchOutput)
{
	TripartitionInitializer tripInit;
	vector<string> names;
	unordered_map<string, int> name2id;
	for (int i = 0; i < options.threads; i++) {
		tripInit.nodes.emplace_back();
		tripInit.leafParent.emplace_back();
	}

	GeneTreeReader reader(tripInit, names, name2id);
	reader.readMapping(options.mapping_path);
	reader.registerSpeciesTreeTaxa(speciesTreeText);
	reader.readGeneTrees(options.input_path, options.seed, options.gene_id_offset);
	if (reader.nGeneTrees == 0) throw runtime_error("No gene trees found in input");

	ThreadPool threadPool;
	threadPool.initialize(options.threads);
	ConstrainedOptimizationAlgorithm alg(
		names.size(), tripInit, names, threadPool, -1
	);
	alg.addGuideTree(speciesTreeText, name2id, -1);
	alg.computeOptimalTree();

	writeGeneStatisticsHeader(branchOutput);
		struct Visitor {
			int nGeneTrees;
			int nTaxa;
			ostream& out;
			int geneIdOffset;

			void operator()(Quadrupartition& quad, const vector<int>& clade) {
				writeGeneStatistics(
					quad,
					nGeneTrees,
					out,
					canonicalSplitKey(clade, nTaxa),
					geneIdOffset
				);
			}
		} visitor = {
			reader.nGeneTrees,
			(int) names.size(),
			branchOutput,
			options.gene_id_offset
		};
	visitBranchesForIILS(alg, visitor);
}

namespace iils {

namespace wire {
// All numbers are little endian; payloads have explicit sizes and completion IDs.
constexpr uint32_t MAGIC = 0x534c4949;
constexpr uint32_t VERSION = 1;

struct Buffer {
	string data;
	size_t pos = 0;
	void put32(uint32_t v) { for (int i=0; i<4; ++i) data.push_back(char(v >> (8*i))); }
	void put64(uint64_t v) { for (int i=0; i<8; ++i) data.push_back(char(v >> (8*i))); }
	void putDouble(double v) { uint64_t bits; memcpy(&bits, &v, 8); put64(bits); }
	void putString(const string& s) { put64(s.size()); data += s; }
	uint64_t get(unsigned n) {
		if (n > data.size()-pos) throw runtime_error("truncated request payload");
		uint64_t v=0;
		for (unsigned i=0; i<n; ++i) v |= uint64_t(uint8_t(data[pos++])) << (8*i);
		return v;
	}
	uint32_t get32() { return get(4); }
	uint64_t get64() { return get(8); }
	double getDouble() { uint64_t bits=get64(); double v; memcpy(&v,&bits,8); return v; }
	string getString() {
		uint64_t n=get64();
		if (n > data.size()-pos) throw runtime_error("truncated request string");
		string s=data.substr(pos,n); pos+=n; return s;
	}
	void finish() { if (pos != data.size()) throw runtime_error("trailing request bytes"); }
};

double historicalPrecision(double x) {
	char buffer[64]; snprintf(buffer, sizeof(buffer), "%.12g", x); return strtod(buffer,nullptr);
}

struct MappedBank {
	int fd=-1;
	const unsigned char* bytes=nullptr;
	size_t size=0;
	uint32_t count=0, taxa=0;
	~MappedBank() { close(); }
	void close() {
		if (bytes) munmap(const_cast<unsigned char*>(bytes),size);
		if (fd>=0) ::close(fd);
		bytes=nullptr; fd=-1; size=0;
	}
	uint64_t number(size_t at,unsigned n) const {
		if (at>size || n>size-at) throw runtime_error("truncated bank");
		uint64_t value=0;
		for (unsigned i=0;i<n;++i) value|=uint64_t(bytes[at+i])<<(8*i);
		return value;
	}
	void open(const string& path, uint32_t expectedCount,uint32_t expectedTaxa) {
		if (bytes) throw runtime_error("bank already initialized");
		fd=::open(path.c_str(),O_RDONLY);
		struct stat st;
		if (fd<0 || fstat(fd,&st)!=0 || st.st_size<24) throw runtime_error("unable to open topology bank");
		size=st.st_size;
		void* mapped=mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0);
		if (mapped==MAP_FAILED) throw runtime_error("unable to mmap topology bank");
		bytes=static_cast<const unsigned char*>(mapped);
		if (memcmp(bytes,"IILSBK01",8)) throw runtime_error("invalid topology bank version");
		count=number(8,4); taxa=number(12,4);
		if (count!=expectedCount || taxa!=expectedTaxa || !count || !taxa)
			throw runtime_error("topology bank identity mismatch");
		uint64_t previous=16+8*(uint64_t(count)+1);
		if (number(16,8)!=previous) throw runtime_error("invalid bank start offset");
		for (uint64_t i=1;i<=count;++i) {
			uint64_t next=number(16+8*i,8);
			if (next<=previous || next>size) throw runtime_error("invalid topology bank offsets");
			previous=next;
		}
		if (previous!=size) throw runtime_error("trailing bank bytes");
	}
	string tree(uint32_t id,const vector<bool>& present,const vector<string>& names) const {
		if (!bytes || id>=count || present.size()!=taxa || names.size()!=taxa)
			throw runtime_error("gene ID or bank taxa out of range");
		uint64_t start=number(16+8*uint64_t(id),8), stop=number(24+8*uint64_t(id),8);
		uint32_t n=number(start,4);
		if (!n || n>2*uint64_t(taxa) || stop-start!=4+12*uint64_t(n))
			throw runtime_error("invalid topology node count");
		// Only requested topologies are decoded. Unary paths are suppressed while pruning.
		vector<string> stack;
		vector<bool> seen(taxa,false);
		for (uint32_t i=0;i<n;++i) {
			int32_t code=static_cast<int32_t>(number(start+4+12*uint64_t(i),4));
			if (code>=0) {
				if (uint32_t(code)>=taxa || seen[code]) throw runtime_error("invalid or duplicate bank leaf");
				seen[code]=true;
				stack.push_back(present[code] ? names[code] : "");
			} else {
				uint64_t children=-int64_t(code);
				if (children<2 || children>stack.size()) throw runtime_error("invalid bank children");
				string joined; size_t kept=0;
				for (size_t k=stack.size()-children;k<stack.size();++k) {
					if (stack[k].empty()) continue;
					if (kept++) joined+=",";
					joined+=stack[k];
				}
				stack.resize(stack.size()-children);
				stack.push_back(kept>1 ? "("+joined+")" : joined);
			}
		}
		if (stack.size()!=1 || stack[0].empty() || find(seen.begin(),seen.end(),false)!=seen.end())
			throw runtime_error("incomplete topology record");
		return stack[0]+";";
	}
};

void lossBatch(Buffer& request,Buffer& response,const StatisticsOptions& options,
               const string& guide, MappedBank& bank,ThreadPool& threads) {
	uint32_t tasks=request.get32();
	if (!tasks || tasks>4096) throw runtime_error("invalid task count");
	response.put32(tasks);
	for (uint32_t task=0;task<tasks;++task) {
		uint64_t taskId=request.get64();
		TripartitionInitializer init;
		// One native thread per persistent process avoids ASTER's per-color thread spawning.
		init.nodes.emplace_back(); init.leafParent.emplace_back();
		vector<string> names; unordered_map<string,int> name2id;
		GeneTreeReader reader(init,names,name2id);
		reader.registerSpeciesTreeTaxa(guide);
		uint32_t nPresent=request.get32();
		if (nPresent<3 || nPresent>names.size()) throw runtime_error("invalid leaf-set size");
		vector<bool> present(names.size(),false);
		for (uint32_t i=0;i<nPresent;++i) {
			uint32_t taxon=request.get32();
			if (taxon>=names.size() || present[taxon]) throw runtime_error("invalid leaf-set ID");
			present[taxon]=true;
		}
		uint32_t nGenes=request.get32();
		if (!nGenes || nGenes>4096) throw runtime_error("invalid gene block size");
		vector<pair<int,string>> records;
		unordered_map<uint32_t,uint32_t> geneIndex;
		for (uint32_t i=0;i<nGenes;++i) {
			uint32_t id=request.get32();
			if (id>INT32_MAX || !geneIndex.emplace(id,i).second) throw runtime_error("invalid or duplicate gene ID");
			records.emplace_back(id,bank.tree(id,present,names));
		}
		struct Target { uint32_t id,denominator; vector<uint32_t> genes; vector<double> loss; unordered_map<string,double> weights; };
		uint32_t nLoci=request.get32();
		if (!nLoci || nLoci>4096) throw runtime_error("invalid locus count");
		vector<Target> targets;
		unordered_set<uint32_t> locusIds;
		vector<vector<pair<uint32_t,uint32_t>>> routes(nGenes);
		for (uint32_t i=0;i<nLoci;++i) {
			Target target;
			target.id=request.get32(); target.denominator=request.get32();
			if (!target.denominator || !locusIds.insert(target.id).second) throw runtime_error("invalid locus identity or denominator");
			uint32_t coefficients=request.get32();
			if (coefficients>names.size()) throw runtime_error("invalid coefficient count");
			for (uint32_t j=0;j<coefficients;++j) {
				string key=request.getString(); double weight=request.getDouble();
				if (!isfinite(weight) || weight<0 || !target.weights.emplace(key,weight).second)
					throw runtime_error("invalid or duplicate coefficient");
			}
			uint32_t requested=request.get32();
			if (!requested || requested>nGenes) throw runtime_error("invalid requested gene count");
			unordered_set<uint32_t> unique;
			for (uint32_t j=0;j<requested;++j) {
				uint32_t id=request.get32(); auto it=geneIndex.find(id);
				if (it==geneIndex.end() || !unique.insert(id).second) throw runtime_error("invalid target gene ID");
				target.genes.push_back(id); routes[it->second].emplace_back(i,j);
			}
			target.loss.assign(requested,0.0); targets.push_back(move(target));
		}
		reader.readGeneTreeRecords(records,options.seed);
		RND_GENERATOR.seed(options.seed); srand(options.seed);
		ConstrainedOptimizationAlgorithm alg(names.size(),init,names,threads,-1);
		alg.addGuideTree(guide,name2id,-1); alg.computeOptimalTree();
		unordered_set<string> visited;
		auto visitor=[&](Quadrupartition& quad,const vector<int>& clade) {
			string key=canonicalSplitKey(clade,names.size());
			if (!visited.insert(key).second) throw runtime_error("duplicate guide split");
			vector<double> weights;
			for (const auto& target:targets) {
				auto it=target.weights.find(key); weights.push_back(it==target.weights.end()?0:it->second);
			}
			auto roots=partitionRoots(quad.parts[0]);
			if (roots.size()!=nGenes) throw runtime_error("incomplete native gene statistics");
			for (uint32_t g=0;g<nGenes;++g) {
				int root=roots[g]; const auto& node=quad.parts[0].nodes[root];
				double capacity=double((long double)node.a*node.b*node.c*node.d);
				double value=capacity>0 ? historicalPrecision((quad.parts[0].score2[root]+quad.parts[0].score3[root])/capacity) : 0;
				if (!isfinite(value) || value<0) throw runtime_error("invalid discordance mass");
				for (const auto& route:routes[g]) targets[route.first].loss[route.second]+=weights[route.first]*value;
			}
		};
		visitBranchesForIILS(alg,visitor);
		response.put64(taskId); response.put32(nGenes); response.put32(nLoci);
		for (const auto& target:targets) {
			for (const auto& weight:target.weights) if (!visited.count(weight.first)) throw runtime_error("unknown split coefficient");
			response.put32(target.id); response.put32(target.genes.size());
			for (size_t j=0;j<target.genes.size();++j) {
				double loss=target.loss[j]/target.denominator;
				if (!isfinite(loss)) throw runtime_error("nonfinite loss");
				response.put32(target.genes[j]); response.putDouble(loss);
			}
		}
	}
	request.finish();
}

void msaBatch(Buffer& request, Buffer& response, const StatisticsOptions& options,
              const string& guide, msa_backend::ThreadPool& threadPool) {
	uint32_t n=request.get32();
	if (!n || n > 1000000) throw runtime_error("invalid MSA batch count");
	msa_backend::TripartitionInitializer init;
	init.nThreads=options.threads;
	vector<string> names;
	unordered_map<string,int> name2id;
	MsaSiteReader reader(init,names,name2id);
	reader.readMapping(options.mapping_path);
	reader.registerSpeciesTreeTaxa(guide);
	unordered_set<uint32_t> ids;
	for (uint32_t i=0; i<n; ++i) {
		uint32_t id=request.get32();
		if (id > INT32_MAX || !ids.insert(id).second) throw runtime_error("invalid or duplicate locus ID");
		string path=request.getString(), raw=request.getString();
		istringstream input(raw);
		FastaAlignment alignment=parseFastaAlignment(input,path);
		reader.readLocus(path,id,options.chunk_size,&alignment);
	}
	request.finish();
	init.nSpecies=names.size();
	msa_backend::RND_GENERATOR.seed(options.seed);
	srand(options.seed);
	msa_backend::ConstrainedOptimizationAlgorithm alg(names.size(),init,names,threadPool,-1);
	alg.addGuideTree(guide,name2id,-1);
	alg.computeOptimalTree();
	Buffer branches;
	uint32_t splitCount=0;
	auto visitor = [&](msa_backend::Quadrupartition& quad, const vector<int>& clade) {
		branches.putString(canonicalSplitKey(clade,names.size()));
		++splitCount;
		for (const auto& locus: reader.loci) {
			auto score=scoreMsaLocusForCurrentColoring(locus,quad);
			for (double v: score) branches.putDouble(historicalPrecision(v));
			branches.putDouble(historicalPrecision(score[0]+score[1]+score[2]));
			branches.put64(msaQuartetCapacityForCurrentColoring(locus,quad));
		}
	};
	visitMsaBranchesForStatistics(alg,visitor);
	response.put32(n); response.put32(splitCount); response.put32(names.size());
	for (const auto& name: names) response.putString(name);
	for (const auto& locus: reader.loci) {
		response.put32(locus.locusId);
		response.put64(locus.nSitesRaw); response.put64(locus.nSitesUsed);
		response.put32(locus.lastGeneBlock-locus.firstGeneBlock);
		response.put32(locus.presentTaxa.size());
		for (int taxon: locus.presentTaxa) response.put32(taxon);
	}
	response.data += branches.data;
}
} // namespace wire

void serve_statistics(const StatisticsOptions& options, istream& input, ostream& output) {
	if (options.threads < 1 || options.chunk_size < 1 || options.seed < 0)
		throw runtime_error("invalid worker configuration");
	LOG.enabled=false; msa_backend::LOG.enabled=false;
	const string guide=stripInternalNodeLabels(readWholeFile(options.constraint_tree_path,"guide"));
	msa_backend::ThreadPool msaThreads(options.threads);
	ThreadPool geneThreads(1);
	wire::MappedBank bank;
	uint64_t lastId=0;
	while (input.peek()!=EOF) {
		wire::Buffer header;
		header.data.resize(28);
		if (!input.read(header.data.data(),28)) throw runtime_error("truncated request header");
		if (header.get32()!=wire::MAGIC || header.get32()!=wire::VERSION)
			throw runtime_error("unsupported worker protocol");
		uint32_t op=header.get32(); uint64_t id=header.get64(), size=header.get64();
		if (id<=lastId || size>(uint64_t(1)<<30)) throw runtime_error("invalid request ID or size");
		lastId=id;
		wire::Buffer request,response;
		request.data.resize(size);
		if (!input.read(request.data.data(),size)) throw runtime_error("truncated request");
		uint32_t status=0;
		try {
			if (op==1) wire::msaBatch(request,response,options,guide,msaThreads);
			else if (op==2) {
				string path=request.getString(); uint32_t count=request.get32(), taxa=request.get32();
				request.finish(); bank.open(path,count,taxa); response.put32(count); response.put32(taxa);
			}
			else if (op==3) wire::lossBatch(request,response,options,guide,bank,geneThreads);
			else throw runtime_error("unsupported request operation");
		} catch (const exception& e) {
			status=1; response.data.clear(); response.putString(e.what());
		}
		wire::Buffer result;
		result.put32(wire::MAGIC); result.put32(wire::VERSION); result.put32(status);
		result.put64(id); result.put64(response.data.size());
		output.write(result.data.data(),result.data.size());
		output.write(response.data.data(),response.data.size()); output.flush();
		if (!output) throw runtime_error("worker output pipe failed");
		if (status) return;
	}
}

void write_statistics(
	const StatisticsOptions& options,
	ostream& branch_output,
	ostream* item_output)
{
	if (options.input_path.empty()) throw runtime_error("input_path is required");
	if (options.constraint_tree_path.empty()) {
		throw runtime_error("constraint_tree_path is required");
	}
	if (options.threads < 1) throw runtime_error("threads must be positive");
	if (options.chunk_size < 1) throw runtime_error("chunk_size must be positive");
		if (options.seed < 0) throw runtime_error("seed must be nonnegative");
		if (options.gene_id_offset < 0) {
			throw runtime_error("gene_id_offset must be nonnegative");
		}
		if (options.input_type == InputType::MsaList && options.gene_id_offset != 0) {
			throw runtime_error("gene_id_offset is only valid for gene-tree input");
		}

	srand(options.seed);
	RND_GENERATOR.seed(options.seed);
	msa_backend::RND_GENERATOR.seed(options.seed);
	LOG.enabled = false;
	msa_backend::LOG.enabled = false;

	string speciesTreeText = stripInternalNodeLabels(
		readWholeFile(options.constraint_tree_path, "species tree")
	);
	if (options.input_type == InputType::MsaList) {
		if (item_output == nullptr) {
			throw runtime_error("MSA statistics require item_output");
		}
		runMsaStatistics(options, speciesTreeText, branch_output, *item_output);
	}
	else {
		runGeneTreeStatistics(options, speciesTreeText, branch_output);
	}
}

}  // namespace iils
