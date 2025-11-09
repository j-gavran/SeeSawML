# Documentation for SeeSawML

Documentation is available at: https://seesawml.docs.cern.ch/.

## Documentation guide

Clone the repository:

```bash
git clone ssh://git@gitlab.cern.ch:7999/atlas-dch-seesaw-analyses/seesaw-ml-docs.git
```

After this step, make sure to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

### Style

Code is documented using the [Numpy Python Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html). Please follow this style guide when documenting code. All the code documentation is automatically extracted and added to the documentation using [mkdocstrings](https://mkdocstrings.github.io/python/) from the submodule.

### MkDocs

[MkDocs](https://www.mkdocs.org/) is a static site generator that's geared towards project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file. All documentation in the source files is added to MkDocs using [mkdocstrings](https://mkdocstrings.github.io/python/). [Material](https://squidfunk.github.io/mkdocs-material/) for MkDocs is used as the theme.

### Building

MkDocs can be installed via pip:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To locally build the documentation, run:

```bash
mkdocs build
```

and to serve the documentation locally, run:

```bash
mkdocs serve -a localhost:<port>
```

### Deployment

The documentation is deployed to GitLab Pages with the configuration in `.gitlab-ci.yml`. Deployment is triggered on every push to any branch. More information can be found [here](https://how-to.docs.cern.ch/).
