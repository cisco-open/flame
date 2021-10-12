# Contributing to Fledge
Thank you for taking time to start contributing! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the Apache License, Version 2
In short, when you submit code changes, your submissions are understood to be under the same [Apache License](LICENSE) that covers the project.
Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/cisco/fledge/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/cisco/fledge/issues).

## Write bug reports with detail, background, and sample code

Please consider to include the following in a bug report:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happened
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style
There is no strict coding style guideline, but some basic suggestions are:

* Try to run `go fmt ./...` and `./lint.sh`
* When go packages are imported in a file, packages are grouped into three categories: go standard packages, third party packages and project's own packages.
Place a new line between each group of packages.

## License
By contributing, you agree that your contributions will be licensed under its Apache License, Version 2.

## References
This document was adapted from [here](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).
