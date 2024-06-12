# chest-xray-detection

## Installation

### Library requirements

- Python 3.10.5
- Poetry 1.8.2


## Pyenv
Some ubuntu installs needed which should be installed before installing `python` with `pyenv`:
```
sudo apt-get install libffi-dev
sudo apt-get install libsqlite3-dev
```

 We use pyenv to manage python, currently version `3.10.5`:
```bash
# download pyenv
curl https://pyenv.run | bash

# install python
pyenv install 3.10.5

# select python
pyenv global 3.10.5
```

## Poetry

Install `Poetry 1.8.2` on your root system
```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2
```

Install all module dependencies inside `pyproject.toml`.

```bash
poetry install

# activate virtual environment
poetry shell
```

**Note** : If you activate your environment within your shell with `poetry shell`, you can execute all your commands directly without specifying `poetry run` first.

Select venv in VSCode located at `/home/ubuntu/.cache/pypoetry/virtualenvs`