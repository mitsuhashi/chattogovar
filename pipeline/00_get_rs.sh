#!/usr/bin/env bash

#
# Create a list of rs_numbers
#

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

# We downloaded https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/mutation2pubtator3.gz 
# at $repo_root/questions/pubtator3/ in December, 2024

cd $repo_root/questions/pubtator3

gzip -dc mutation2pubtator3 | sort -nr | grep rs | head -31 | sed -E 's/rs0+/rs/' > pubtator_rs_latest_31.txt || true
cut -f 3 pubtator_rs_latest_31.txt | sort -u > rs_30.txt

#Post-processing
#	1.	Search dbSNP and record the corresponding gene symbol for each rs ID in the second column.
#	2.	rs1235072590 has been merged into rs80356821, so replace rs1235072590 with rs80356821.
#	•	https://www.ncbi.nlm.nih.gov/snp/rs1235072590
#	•	Retrieve the gene symbol “HBB” from https://www.ncbi.nlm.nih.gov/snp/rs80356821
#	3.	Save the final file as rs_30_with_symbol.txt with two columns: rs ID and gene symbol, separated by a tab.
