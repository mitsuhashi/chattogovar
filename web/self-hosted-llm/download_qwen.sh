#!/usr/bin/env bash
#
# download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF from huggingface
#
huggingface-cli download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF --include "Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf" --local-dir ./models --local-dir-use-symlinks False
