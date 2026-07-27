#!/bin/bash
# Safety wrapper for SWE-bench execution
# Restricts Shepherd to only operate within the specified working directory

WORK_DIR="$1"
shift  # Remove first argument, rest are the shepherd command

# Validate work directory exists and is in /tmp
if [[ ! -d "$WORK_DIR" ]]; then
    echo "ERROR: Work directory does not exist: $WORK_DIR"
    exit 1
fi

if [[ "$WORK_DIR" != /tmp/swebench_* ]]; then
    echo "ERROR: Work directory must be in /tmp/swebench_*: $WORK_DIR"
    exit 1
fi

# Change to work directory
cd "$WORK_DIR" || exit 1

# Set restrictive umask
umask 077

# Set HOME to work directory (prevents accessing real home)
export HOME="$WORK_DIR"

# Unset dangerous environment variables
unset SSH_AUTH_SOCK
unset SSH_AGENT_PID
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY

# Run shepherd with remaining arguments
exec "$@"
