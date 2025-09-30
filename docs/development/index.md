# Development

For the development of the `commonroad-stl-monitor` we recommend [python 3.10](https://www.python.org/downloads/) and [poetry](https://www.python.org/downloads/).

To get started with development, install the necessary dependencies:

```sh
$ poetry install --extras test --extras dev --extras visualization --extras docs
```

Additionally, you can install extras like `visualization` to enable advanced AST visualizations. The visualization requires a working `graphiz` installation, which you should be able to source from your distros package registry.

## Tests

To run the tests you can use:

```sh
$ poetry run pytest
# Run the tests with coverage
$ poetry run coverage run -m pytest
# Report the coverage on the command line
$ poetry run coverage report
# Alternativly generate HTML coverage files
$ poetry run coverage html
```

## pre-commit

pre-commit is used to run a variety of tools and checks on the code (e.g. formatting with `ruff`).
You can install the hooks with:

```sh
$ pre-commit install
```

Those hooks will be run every time you run `git commit`.
Alternatively you can execute the pre-commit hooks manually:

```sh
$ pre-commit run --all-files
```


## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/). Before building, make sure to install the required dependencies:

```sh
$ poetry install --extras docs
```

Then you can build the documentation with:

```sh
$ poetry run mkdocs build
```

While writing/developing the documentation you can run a hot-reloading webserver:

```sh
$ poetry run mkdocs serve
```
