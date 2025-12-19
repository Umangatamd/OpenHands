#!/bin/bash
set -e

# Test the kernel optimization tools with OpenHands

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/test_kernel_tools_config.toml"
TASK_FILE="${SCRIPT_DIR}/kernel_optimization_task.txt"

echo "========================================"
echo "Testing Kernel Optimization Tools"
echo "========================================"
echo "Config: ${CONFIG}"
echo "Task: ${TASK_FILE}"
echo "Started at: $(date)"
echo "========================================"

# Ensure we're using the OpenHands-dev version
cd "${SCRIPT_DIR}"

# Set environment variables for LLM calls
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export EVAL_LLM_MODEL="${EVAL_LLM_MODEL:-gpt-4o-mini}"

# Check if API key is set
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Kernel evaluation tools will not work."
    echo "Set it with: export OPENAI_API_KEY='your-key'"
fi

# Run OpenHands
python -u -m openhands.core.main \
    --config "${CONFIG}" \
    -f "${TASK_FILE}" \
    2>&1 | tee "${SCRIPT_DIR}/kernel_tools_test.log"

echo "========================================"
echo "Complete at $(date)"
echo "Log saved to: ${SCRIPT_DIR}/kernel_tools_test.log"
echo "========================================"

