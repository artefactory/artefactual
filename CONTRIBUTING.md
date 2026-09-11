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

There is no tag to push and no version to edit: the version lives in the git tag, and the
tag is created by CI. Two things start it.

**Land the changes, then say when.** Nothing about a merge releases by itself, so the
pull requests go in normally and the release is a separate decision, taken once they are
all on `main`:

```bash
gh workflow run release.yaml --ref main
```

Starting a workflow needs write access and the `pypi` environment still holds its required
reviewer, so this is a way to ask rather than a way around the gate. It refuses any ref but
`main`: the tag is created on `HEAD`, and a tag on a side branch is not reachable by
`git describe` from `main`, which is where `hatch-vcs` reads the version back from. A
`HEAD` that already carries a tag is refused too — that commit has been released, and a
second tag on it would spend a version number on an identical tree.

**Or decide at merge time**, with the **`release`** label on the pull request. A merge to
`main` releases when the pull request it came from carried the label, and does nothing
otherwise:

```bash
gh pr edit <number> --add-label release
```

This is the shorter path when one pull request is the whole release. It is also the one
that has to be remembered in the middle of merging, and the label is not visible in the
diff being reviewed, so prefer it for a single change and dispatch for a batch.

Ordinary merges publish nothing, so a fix to a fix does not spend a version number. A
commit pushed straight to `main`, belonging to no pull request, never releases.

Do not start a dispatch while a labelled merge is still releasing: the two are serialised
rather than rejected, so the second would run against whatever `main` is by then. The tag
job refuses an already-released `HEAD`, which catches the common case, but the Actions tab
is the thing to check first.

`bump-my-version` computes the next tag from the most recent reachable one and creates it,
configured to write no files and make no commit. `hatch-vcs` then reads the version back
off that tag at build time, so what is tagged and what is built cannot disagree — and
`pyproject.toml` carries no version string to fall out of step.

The chain runs in one workflow, because a tag pushed with `GITHUB_TOKEN` does not start a
workflow run: chaining on the tag would leave the tag created and nothing built.

    `gh workflow run release.yaml --ref main`, or a merge to main
      -> gate             a dispatch is the decision itself; a merge releases only if
                          the pull request it came from carried `release`. Refuses a
                          HEAD that already carries a tag, and names the version this
                          release will be -- computed, not written down anywhere
      -> tests            the suite, against the exact commit being released
      -> docs-build       the site, with its notebooks executed -- a published example
                          that stopped running stops the release here. The pages are
                          told the version the gate named, since the tag does not
                          exist yet
      -> tag              bump-my-version creates that same version as vYYYY.MM.PATCH
      -> build            hatch-vcs derives the version from the tag; the distributions
                          are checked and the wheel is smoke-tested
      -> publish-testpypi uploaded, then checked against the metadata the index serves
      -> publish          PyPI, held for a required reviewer
      -> github-release   the Release, once PyPI has the version
      -> docs-deploy      the pages built above, added to the archive of previous
                          releases under their own version directory, and published
                          as the whole site

Everything that can fail without leaving a trace runs before the tag; the tag is the last
recoverable step, and the PyPI upload is the first irreversible one. So a failing test or a
notebook that stopped running costs nothing but a re-run.

The `pypi` environment has a required reviewer, so nothing reaches PyPI unattended. A
release that should not go out is declined there; the tag is already created by then, and a
tag is cheap to delete -- `git push origin :vYYYY.MM.PATCH` -- where a PyPI version is not
reusable.

The site keeps every release. Each one is published under its own version directory, with
`stable/` a copy of the newest and a switcher in the navbar to move between them, so a
reader pinned to an older version still has documentation that matches what they installed.
The archive lives on the `gh-pages` branch because `actions/deploy-pages` replaces the whole
site on every run; that branch is rewritten as a single commit each time, so the repository
grows with the number of versions kept online rather than the number of deploys. At roughly
10 MB a version, prune the oldest directories from `gh-pages` if the site approaches the
1 GB Pages limit.

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

## The example notebooks

The examples under `docs/examples/` are MyST Markdown, not `.ipynb`. The source is code and
prose: no stored outputs, no execution counts, no base64 images, so a change to one is
reviewable as an ordinary diff. The `.ipynb` a reader downloads or opens in Colab is written
by the documentation build, into the site, and never committed -- `docs/examples/*.ipynb` is
ignored for that reason.

To work on one in Jupyter, convert it and convert it back:

```bash
uv run jupytext --to ipynb docs/examples/train_wepr.md   # edit train_wepr.ipynb in Jupyter
uv run jupytext --to md:myst docs/examples/train_wepr.ipynb
```

Commit only the `.md`. Pre-commit pipes its code cells through the same `ruff` as the rest
of the repository, so formatting stays consistent without nbQA unpacking a notebook first.

The release build runs every example and publishes what it got, so the pages carry real
outputs while the repository carries none. An example that needs credentials opts out in its
own front matter (`mystnb: {execution_mode: "off"}`) -- which is also why a documentation
preview on a pull request shows the code with nothing under it: that build executes nothing.
