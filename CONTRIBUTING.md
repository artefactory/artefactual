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

Releases are triggered by pushing a version tag.

1. **Bump version (creates commit + tag):**
    ```bash
    # For a patch release (e.g., 2026.01.0 -> 2026.01.1)
    uvx bump-my-version bump patch

    # For a new month's release (e.g., 2025.12.5 -> 2026.01.0)
    uvx bump-my-version bump release

    ```

    * `patch`: Same month → increment patch (2026.01.0 → 2026.01.1)
    * `release`: New month → reset patch (2025.12.5 → 2026.01.0)


2. **Push branch for PR (optional, for testing):**
    ```bash
    git push origin your-branch-name
    gh pr create --title "chore(release): X.Y.Z" --body "Version bump"

    ```


3. **Push the version tag:**
    ```bash
    git push origin vX.Y.Z

    ```

    The tag push triggers the release workflow which:

    * Validates version in code matches the tag
    * Generates changelog with git-cliff
    * Builds the package
    * Creates GitHub release with artifacts
    * Triggers PyPI publishing

### Local Changelog Preview

To preview what will be in the next release notes (grouped by the commit types defined above):

```bash
uvx git-cliff --unreleased

```

### Event-Specific Tags

CalVer tags (e.g., `2026.01.0`) coexist with event-specific tags (e.g., `ECIR2026`). You can create event tags manually:

```bash
git tag ECIR2026
git push origin ECIR2026

```

Both tag types can point to the same commit.
