#!/bin/bash
# Script to commit all new/modified/deleted files and push to the current branch

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

# Detect current branch
branch=$(git rev-parse --abbrev-ref HEAD)

echo "📌 Current branch: $branch"

# Stage all changes (new, modified, deleted)
git add -A

# Commit message
if [ -z "$1" ]; then
  echo "Enter commit message: "
  read commit_msg
else
  commit_msg="$1"
fi

# If nothing to commit, exit
if git diff --cached --quiet; then
  echo "✅ Nothing to commit."
  exit 0
fi

# Commit and push
git commit -m "$commit_msg"
git push origin "$branch"
