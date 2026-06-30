# Repository Agent Instructions

## Git workflow

- When updating this branch against the target/base branch, use rebase rather than merge.
- Prefer `git fetch origin` followed by `git rebase origin/main` (or the current PR base branch if it is not `main`).
- Do not create merge commits just to synchronize with the base branch.
- If a rebased branch has already been pushed, update it with `git push --force-with-lease` rather than `git push --force`.
