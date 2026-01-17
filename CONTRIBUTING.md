# Contributing to Artefactual

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
   - `patch`: Same month → increment patch (2026.01.0 → 2026.01.1)
   - `release`: New month → reset patch (2025.12.5 → 2026.01.0)

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
- Validates version in code matches the tag
- Generates changelog with git-cliff
- Builds the package
- Creates GitHub release with artifacts
- Triggers PyPI publishing

### Local Changelog Preview

To preview what will be in the next release notes:

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
