"""Write the version-switcher index for the published documentation.

The site is one directory per release plus a `stable` copy of the newest. This reads the
directories that are actually there and writes the JSON the theme's dropdown fetches, so
the dropdown cannot advertise a version that was never published, or omit one that was.

Usage: build_switcher.py index <site-root> <base-url>
       build_switcher.py canonical <version>
"""

import json
import re
import sys
from pathlib import Path

RELEASE_DIR = re.compile(r"\d{4}\.\d+\.\d+")


def canonical_version(raw: str) -> str:
    """The version as the installed package reports it.

    Two spellings of one release are in play: the tag and `bump-my-version` write the
    zero-padded CalVer month (2026.09.0), while PEP 440 normalisation strips the padding,
    so `artefactual.__version__` -- and therefore the `version_match` the theme compares
    against -- is 2026.9.0. The site must use the second, or the dropdown never marks the
    version being read as the current one.
    """
    return ".".join(str(int(part)) for part in raw.split("."))


def released_versions(site: Path) -> list[str]:
    """The release directories under `site`, newest first.

    Ordered numerically rather than lexically: sorting as strings puts 2026.10.0 before
    2026.9.0, which would offer the wrong version as stable.
    """
    names = [d.name for d in site.iterdir() if d.is_dir() and RELEASE_DIR.fullmatch(d.name)]
    return sorted(names, key=lambda n: [int(p) for p in n.split(".")], reverse=True)


def switcher_entries(versions: list[str], base_url: str) -> list[dict[str, object]]:
    """The dropdown's contents: the newest release as `stable`, then the older ones.

    The newest appears once, under the `stable` URL rather than its own. Listing it twice
    would give two entries the same `version`, and the theme marks the first one matching
    the page's version as current -- so the duplicate is never reachable as "current" and
    only makes the dropdown longer.
    """
    base = base_url.rstrip("/")
    newest, older = versions[0], versions[1:]
    entries: list[dict[str, object]] = [
        {"name": f"{newest} (stable)", "version": newest, "url": f"{base}/stable/", "preferred": True}
    ]
    entries += [{"name": v, "version": v, "url": f"{base}/{v}/"} for v in older]
    return entries


def write_index(site_root: str, base_url: str) -> None:
    site = Path(site_root)
    versions = released_versions(site)
    if not versions:
        msg = f"no release directories under {site}; nothing to offer in the switcher"
        raise SystemExit(msg)

    entries = switcher_entries(versions, base_url)
    # The keys the theme requires. It would have checked them itself by fetching the file at
    # build time, which this site cannot rely on -- the file is written here, after the build.
    missing = [e for e in entries if not {"version", "url"} <= set(e)]
    if missing:
        msg = f"switcher entries missing required keys: {missing}"
        raise SystemExit(msg)

    (site / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"{len(versions)} version(s), stable = {versions[0]}")


if __name__ == "__main__":
    verb, *rest = sys.argv[1:]
    if verb == "canonical":
        print(canonical_version(*rest))
    elif verb == "index":
        write_index(*rest)
    else:
        msg = f"unknown verb {verb!r}; expected 'index' or 'canonical'"
        raise SystemExit(msg)
