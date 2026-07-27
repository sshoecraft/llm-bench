#!/usr/bin/env python3
"""
Simple test script to run open-codex on a single SWE-bench task manually.

This helps debug the interaction before building full automation.

Usage:
    ./test_one_swebench.py [task_number]
"""

import subprocess
import os
import sys
import tempfile
import json

# SWE-bench repos
REPOS = {
    "django/django": "https://github.com/django/django.git",
    "psf/requests": "https://github.com/psf/requests.git",
}

# Sample easy tasks (you can replace with real ones)
SAMPLE_TASKS = [
    {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "main",  # Replace with actual commit
        "problem_statement": """
The admin list view doesn't properly escape HTML in readonly fields.

Steps to reproduce:
1. Create a model with a readonly field containing HTML
2. View the admin list page
3. HTML is rendered instead of escaped

Expected: HTML should be escaped
Actual: HTML is rendered as-is
""",
        "test_patch": "",  # Optional test patch
        "FAIL_TO_PASS": "tests.admin_views.test_readonly_fields",
    }
]

def run_codex_on_task(task_idx=0):
    """Run open-codex on a single SWE-bench task"""

    if task_idx >= len(SAMPLE_TASKS):
        print(f"Task {task_idx} not found. Only {len(SAMPLE_TASKS)} sample tasks available.")
        return 1

    task = SAMPLE_TASKS[task_idx]

    print(f"\n{'='*70}")
    print(f"Testing open-codex on: {task['instance_id']}")
    print(f"{'='*70}\n")

    # Create temp directory
    work_dir = tempfile.mkdtemp(prefix=f"codex_test_{task['instance_id']}_")
    print(f"Work directory: {work_dir}")

    try:
        # Get repo
        repo_name = task['repo']
        if repo_name not in REPOS:
            print(f"ERROR: Unknown repo {repo_name}")
            return 1

        repo_url = REPOS[repo_name]
        repo_dir = os.path.join(work_dir, repo_name.split('/')[-1])

        # Clone
        print(f"\nCloning {repo_url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", task['base_commit'], repo_url, repo_dir],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"Clone failed: {result.stderr}")
            return 1

        print("Cloned successfully!")

        # Create prompt file
        prompt = f"""Fix the following issue in the codebase:

{task['problem_statement']}

Use the available tools to:
1. Read relevant files
2. Understand the issue
3. Make the necessary changes
4. Test the fix

When done, type 'q' or send Ctrl+D to exit.
"""

        prompt_file = os.path.join(work_dir, "prompt.txt")
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        print(f"\n{'='*70}")
        print("Starting open-codex...")
        print(f"{'='*70}\n")
        print(f"Work directory: {repo_dir}")
        print(f"\nPrompt:\n{prompt}\n")
        print(f"{'='*70}\n")

        # Set up environment with OLLAMA_BASE_URL
        env = os.environ.copy()
        env['OLLAMA_BASE_URL'] = 'http://localhost:8000/v1'
        env['OPENAI_API_KEY'] = 'dummy-key-for-local-server'  # Required even for ollama provider

        print(f"Environment variables:")
        print(f"  OLLAMA_BASE_URL={env.get('OLLAMA_BASE_URL')}")
        print(f"  OPENAI_API_KEY={'set' if env.get('OPENAI_API_KEY') else 'not set'}")
        print()

        # Run open-codex in quiet mode (non-interactive)
        # -q mode exits after completion
        print("Starting open-codex in quiet + full-auto mode...")
        print(f"Command: open-codex -q --full-auto '<prompt>'\n")

        # Run with -q (quiet) and --full-auto to avoid interactive UI and approval prompts
        result = subprocess.run(
            ["open-codex", "-q", "--full-auto", "--provider", "ollama", "--model", "qwen3-coder", prompt],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)

        print(f"\n{'='*70}")
        print(f"open-codex exited with code: {result.returncode}")
        print(f"{'='*70}\n")

        # Show what files changed
        print("Checking for changes...")
        diff_result = subprocess.run(
            ["git", "diff"],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )

        if diff_result.stdout.strip():
            print("\nChanges made:")
            print(diff_result.stdout[:1000])  # First 1000 chars
            if len(diff_result.stdout) > 1000:
                print(f"\n... ({len(diff_result.stdout) - 1000} more characters)")
        else:
            print("\nNo changes detected.")

        print(f"\nWork directory preserved at: {work_dir}")
        print("You can inspect the results manually.")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    task_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sys.exit(run_codex_on_task(task_idx))
