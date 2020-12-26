Release Process
===============

This document describes the current release process. It may or may not change in the future.

Process
-------

Each week on Monday if there was a commit since the last release create a new release. The version number of this release is in `pyproject.toml`.

To make a release the following things need to happen:

1. Update the `changelog.md` unreleased section to contain the version number and date the release you are building. Create a pull request with the change.
2. Tag the release commit with the version number as soon as the PR is merged.
3. Build the release with `poetry build` and publish it with `poetry publish`
4. Create an entry in GitHub releases with the release notes for the previously tagged commit.

After the release
-----------------

Create a pull request which contains the following changes:

1. Increase the minor version in `pyproject.toml` by one.
2. Update all files which contain the current version number if necessary.
3. Add an unreleased section in the changelog.rst.
