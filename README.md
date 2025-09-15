# Operation Safe Passage

![Coverage](https://vtnsi.github.io/operation_safe_passage/public/coverage.svg)

Read full documentation [here](vtnsi.github.io/operation_safe_passage)

-----

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Authors](#authors)
- [License](#license)

## Installation

Setup the Python environment

```bash
$ python -m venv venv
$ source ./venv/bin/activate
$ pip install .
```

## Quickstart

### Generate Map

Before running the reinforcement learning algorithm, we must first create the map in the form of a network. A MapGenerator object reads in the configs/params.json by default, and it outputs a network.json file with the created nodes and edges.

The MapGenerator is non-deterministic when given the same params.json file. Modifications to the params.json file can be made to construct different environment sizes, terrains, and arguments

```bash
$ python main.py map
$ mv output/network.json config/
```

### Run RL Algorithm

With the environment network created, we can now run the reinforcement learning algorithm

```bash
$ python main.py rl
```

## Authors

* [Stephen Adams](https://nationalsecurity.vt.edu/personnel-directory/adams-stephen.html)
* [Sami Saliba](https://nationalsecurity.vt.edu/personnel-directory/saliba_sami.html)
* [Michael "Alex" Kyer](https://nationalsecurity.vt.edu/personnel-directory/kyer-alex.html)
* [Dan Sobien](https://nationalsecurity.vt.edu/personnel-directory/sobien-daniel.html)
* [Dan DeCollo](https://nationalsecurity.vt.edu/personnel-directory/decollo-dan.html)

## License

`operation_safe_passage` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.