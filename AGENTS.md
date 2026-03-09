# Agent Guidelines

- Do not delete comments or refactor code, unless it is objectively wrong.
- Minimize addition of excess conditionals like None checks if it isn't adding value to the codebase. No usage/ incomplete or undersired result. When in doubt, confirm with yes, no questions.
- Assume that code is being run only on a GPU env. No need to check for torch.cuda.is_available() or similar checks.
