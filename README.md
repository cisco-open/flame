<p align="center"><img src="docs/images/logo.png" alt="flame logo" width="200"/></p>

[![](https://img.shields.io/badge/Flame-Join%20Slack-brightgreen)](https://join.slack.com/t/flame-slack/shared_invite/zt-1mprreo9z-FmpGb1UPi43JOFJKyhIqAQ)

[:fire: **Quickstart! (Ubuntu)** :fire:](docs/quickstart-ubuntu.md)

Flame is a platform that enables developers to compose and deploy federated learning (FL) training workloads easily.
The system is comprised of a service (control plane) and a python library (data plane).
The service manages machine learning workloads, while the python library facilitates composition of ML workloads.
And the library is also responsible for executing FL workloads.
With extensibility of its library, Flame can support various experimentations and use cases.

## Getting started
The target runtime environment is Linux. Development has been mainly conducted under macOS environment.
One should first set up a development environment.
For more details, refer to [here](docs/02-getting-started.md).

Then, users can use a dev/test environment in a single machine on top of minikube.
The detailed instructions are found [here](docs/03-fiab.md).

This repo has the following directory structure:
```
flame
 ├── CODE_OF_CONDUCT.md
 ├── CONTRIBUTING.md
 ├── LICENSE
 ├── Makefile -> build/Makefile
 ├── README.md
 ├── api (specification of REST API for flame apiserver)
 ├── build (configuration files for building flame binaries and container image)
 ├── cmd (source files for flame control plane)
 ├── docs (document folder)
 ├── examples (example folder)
 ├── fiab (dev/test env in a single box)
 ├── go.mod
 ├── go.sum
 ├── lib (python library for core flame data plane)
 ├── lint.sh
 ├── pkg (go packages for cmd)
 └── scripts (utility scripts)
```

## Supported Algorithms/Mechanisms

| Method          | Note                                                                                                                               |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| FedAvg          | https://arxiv.org/pdf/1602.05629.pdf                                                                                               |
| FedYogi         | https://arxiv.org/pdf/2003.00295.pdf                                                                                               |
| FedAdam         | https://arxiv.org/pdf/2003.00295.pdf                                                                                               |
| FedAdaGrad      | https://arxiv.org/pdf/2003.00295.pdf                                                                                               |
| FedProx         | https://arxiv.org/pdf/1812.06127.pdf                                                                                               |
| FedBuff         | Asynchronous FL (https://arxiv.org/pdf/2106.06639.pdf and https://arxiv.org/pdf/2111.04877.pdf); secure aggregation is excluded    |
| FedDyn          | https://arxiv.org/pdf/2111.04263.pdf                                                                                               |
| OORT            | https://arxiv.org/pdf/2010.06081.pdf; client selection algorithm / mechanism; experimental release                                 |
| Hierarchical FL | https://arxiv.org/pdf/1905.06641.pdf; a simplified version where k<sub>2</sub> = 1; support both synchronous and asynchronous FL   |
| Hybrid FL       | A hybrid approach to combine federated learning with ring-reduce; topology motivated from https://openreview.net/pdf?id=H0oaWl6THa |


## Documentation

A full document can be found [here](docs/README.md). The document will be updated on a regular basis.

## Support

We welcome feedback, questions, and issue reports.

* Maintainers' email address: <flame-github-owners@cisco.com>
* [GitHub Issues](https://github.com/cisco-open/flame/issues/new/choose)

## Citation

```
@misc{flame2023,
      title={Federated Learning Operations Made Simple with Flame}, 
      author={Harshit Daga and Jaemin Shin and Dhruv Garg and Ada Gavrilovska and Myungjin Lee and Ramana Rao Kompella},
      year={2023},
      eprint={2305.05118},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
