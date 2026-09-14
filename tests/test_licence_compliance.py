# [AI-GEN] agent=Claude date=2026-08-09 task=Guard the upstream licence texts, which were deleted in a docs cleanup
# modified: [AI-GEN] agent=Claude date=2026-09-12 task=Also guard against the directory being untracked/gitignored (2026-08-15 recurrence)
# reviewed-by: PENDING

"""Licence compliance (AI_RULES.md §5/§7; CLAUDE.md §7; PRD.md §6).

`THIRD_PARTY_LICENSES/` was deleted on 2026-08-09 during a docs cleanup and restored
the same day. Nothing caught it, because nothing imported it — a directory that only
documentation references can vanish silently and stay vanished until a reviewer
notices, which for a licence obligation is the worst possible time.

Both upstreams are MIT. MIT requires the copyright and permission notice to travel with
"all copies or substantial portions of the Software", and this repository adapts the
pruning mathematics from `saediag.pruning` and wraps `circuit-tracer`. The camera-ready
artifact ships that code, so these notices must ship with it. That is a licence
obligation, not a checklist item.
"""

import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LICENCES = REPO / "THIRD_PARTY_LICENSES"

REQUIRED = {
    "circuit-tracer-LICENSE.txt": "Michael Hanna",
    "sae-pruning-paper-LICENSE.txt": "Héctor Borobia",
    # 2026-09-14: src/tasks/ ports the IOI and greater-than generators from ACDC, whose
    # IOI file is itself an edit of Easy-Transformer. Both are MIT.
    "automatic-circuit-discovery-LICENSE.txt": "Arthur Conmy",
    "easy-transformer-LICENSE.txt": "neelnanda-io",
}


class TestUpstreamLicencesAreShipped:
    def test_the_directory_exists(self):
        assert LICENCES.is_dir(), (
            "THIRD_PARTY_LICENSES/ is missing. Both upstreams are MIT and their notices "
            "must ship with the adapted code (CLAUDE.md §7, PRD.md §6). Restore with: "
            "cp ../circuit-tracer-0.5.2/LICENSE THIRD_PARTY_LICENSES/circuit-tracer-LICENSE.txt"
        )

    @pytest.mark.parametrize("filename,holder", sorted(REQUIRED.items()))
    def test_each_licence_is_present_and_names_its_holder(self, filename, holder):
        path = LICENCES / filename
        assert path.is_file(), f"{filename} is missing from THIRD_PARTY_LICENSES/"
        text = path.read_text(encoding="utf-8", errors="replace")
        assert holder in text, f"{filename} does not name {holder} — is it the right file?"

    @pytest.mark.parametrize("filename", sorted(REQUIRED))
    def test_each_licence_carries_the_permission_notice(self, filename):
        """The clause MIT actually obliges us to reproduce."""
        text = (LICENCES / filename).read_text(encoding="utf-8", errors="replace")
        assert "Permission is hereby granted" in text
        assert "WITHOUT WARRANTY OF ANY KIND" in text.upper()

    def test_the_provenance_readme_exists(self):
        assert (LICENCES / "README.md").is_file(), (
            "THIRD_PARTY_LICENSES/README.md records which fork each text came from"
        )


class TestOurOwnLicence:
    def test_the_project_licence_exists(self):
        assert (REPO / "LICENSE").is_file(), "PRD.md §6 requires the artifact to carry a licence"

    def test_it_is_mit_which_is_compatible_with_both_upstreams(self):
        """MIT-on-MIT is compatible provided the upstream notices travel along."""
        text = (REPO / "LICENSE").read_text(encoding="utf-8", errors="replace")
        assert "MIT License" in text
        assert "Permission is hereby granted" in text


class TestTheLicencesAreActuallyTrackedByGit:
    """The 2026-08-15 recurrence: files on disk, but invisible to the repository.

    The tests above pass whenever the texts exist locally, which is exactly the state
    a working machine is in after someone gitignores the directory — green suite,
    broken clone. `f8bd168` deleted the directory and `29d7ed3` added it to
    .gitignore, so the notices were absent from every clone for 3 weeks while local
    runs stayed green. Deletion is caught above; *untracking* is caught here.
    """

    @staticmethod
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    @pytest.fixture(autouse=True)
    def _requires_git_repo(self):
        if shutil.which("git") is None:
            pytest.skip("git not on PATH")
        if self._git("rev-parse", "--git-dir").returncode != 0:
            pytest.skip("not a git working tree (source distribution?)")

    @pytest.mark.parametrize("filename", sorted(REQUIRED) + ["README.md"])
    def test_each_file_is_tracked(self, filename):
        rel = f"THIRD_PARTY_LICENSES/{filename}"
        tracked = self._git("ls-files", "--error-unmatch", rel).returncode == 0
        assert tracked, (
            f"{rel} exists on disk but git does not track it, so it is missing from "
            "every clone and from the artifact. Run: git add -f " + rel
        )

    def test_the_directory_is_not_gitignored(self):
        """`git check-ignore` exits 0 when a path IS ignored.

        Two flags of subtlety, both found by testing this guard rather than trusting it
        (git 2.51.1):

        `--no-index` is essential. Without it git declines to report a *tracked* path as
        ignored, so a live .gitignore rule stays invisible until the day someone deletes
        the directory — precisely when it does the damage. We assert no such rule
        exists, not merely that today's files escaped one.

        The path must carry **no trailing slash**. With `--no-index`, any path written
        `foo/` matches a blank line in .gitignore and reports ignored; `docs/` and `src/`
        both do it. Slash-free is accurate here because the directory exists on disk
        (asserted above), so git can tell a directory-only pattern applies to it. In the
        one case where that inference could fail — directory deleted *and* ignored —
        test_the_directory_exists and test_each_file_is_tracked both fail anyway.
        """
        ignored = self._git(
            "check-ignore", "--no-index", "-q", "THIRD_PARTY_LICENSES"
        ).returncode == 0
        assert not ignored, (
            "THIRD_PARTY_LICENSES/ is matched by a .gitignore rule. MIT obliges the "
            "upstream notices to ship with the artifact; an ignored directory silently "
            "will not. Remove the rule rather than relying on `git add -f`."
        )
