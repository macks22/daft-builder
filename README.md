# daft-builder

Wrapper library on daft that provides a builder interface for rendering probabilistic graphical models (PGMs).

See `notebooks/examples` for example usage.

## Building and releasing

To build a wheel, run:

`python setup.py bdist_wheel`

The wheel will then be found in the `dist` directory. A similar command will build a source distribution:

`python setup.py sdist`

To release to PyPi:

`twine upload dist/*`

If creating a release, you'll want to make sure you tag the stuff you're building. First run `git tag --list` to see the most recent tag. Then choose an appropriate semantic versioning increment and run `git tag <new_tag>` to add this tag. Finally, push up the new tag to GitHub by running `git push origin --tags`.

To release on GitHub, use the GUI, following [this guide](https://help.github.com/en/articles/creating-releases).
