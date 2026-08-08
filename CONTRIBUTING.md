# Contributing to Artefactual

## Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This leads to more readable messages that are easy to follow when looking through the project history, and allows us to automatically generate changelogs.

### Commit Structure

Each commit message should be structured as follows:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

```

### Allowed Types

The `<type>` must be one of the following:

* **build**: Changes that affect the build system or external dependencies
* **ci**: Changes to our CI configuration files and scripts
* **chore**: Changes to the build process or auxiliary tools
* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* **test**: Adding missing tests or correcting existing tests

### Semantic Versioning & Impact

Following the Conventional Commits specification:

1. **fix:** a commit of the *type* `fix` patches a bug in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** a commit of the *type* `feat` introduces a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any *type*.
4. **Other types:** Types other than `fix:` and `feat:` are allowed (e.g., `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`).
5. **Footers:** Footers other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

---

## Release Workflow

This project uses [CalVer](https://calver.org/) versioning with the format `YYYY.MM.PATCH` (e.g., `2026.01.0`).

### Creating a Release

A merge to `main` releases when its pull request carried the **`release`** label, and does
nothing otherwise. There is no tag to push and no version to edit: the version lives in the
git tag, and the tag is created by CI.

Label the pull request before merging it:

```bash
gh pr edit <number> --add-label release
```

Ordinary merges publish nothing, so a fix to a fix does not spend a version number. A
commit pushed straight to `main`, belonging to no pull request, never releases.

`bump-my-version` computes the next tag from the most recent reachable one and creates it,
configured to write no files and make no commit. `hatch-vcs` then reads the version back
off that tag at build time, so what is tagged and what is built cannot disagree — and
`pyproject.toml` carries no version string to fall out of step.

The chain runs in one workflow, because a tag pushed with `GITHUB_TOKEN` does not start a
workflow run: chaining on the tag would leave the tag created and nothing built.

    merge to main
      -> gate             is the merged pull request labelled `release`?
      -> tests            the suite, against the exact commit being released
      -> tag              bump-my-version creates vYYYY.MM.PATCH
      -> build            hatch-vcs derives the version; the distributions are checked
                          and the wheel is smoke-tested
      -> publish-testpypi uploaded, then checked against the metadata the index serves
      -> publish          PyPI, held for a required reviewer
      -> github-release   the Release, once PyPI has the version

The `pypi` environment has a required reviewer, so nothing reaches PyPI unattended. A
release that should not go out is declined there; the tag is already created by then, and a
tag is cheap to delete where a PyPI version is not reusable.

Uploading before announcing is deliberate: a Release created first would advertise a
version that a failed upload never produced, under a tag that cannot be reissued. The
upload job holds only the credential to publish and no write access to the repository;
creating the Release is a separate job holding the reverse.

Which increment is taken comes from `bump-my-version`'s configuration, following
[CalVer](https://calver.org/):

* `patch` — same month, increment the patch (2026.01.0 -> 2026.01.1)
* `release` — a new month, reset the patch (2025.12.5 -> 2026.01.0)

To see what the next tag would be without creating it:

```bash
uvx bump-my-version show-bump
```

Every pull request runs the same build the release does, with publishing switched off, so a
packaging fault surfaces before the merge rather than after the tag exists.
