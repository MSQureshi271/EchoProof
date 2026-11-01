"""Capture bounded publisher documents only; never download audio archives."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import uuid

HF_REVISION = "9143e5ea709575ebab6bec52840a1043aada7bb1"


def capture(url, destination, limit=4_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": "EchoProof-source-evidence/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Publisher document exceeds configured byte limit")
        final_url = response.geturl()
    destination.write_bytes(body)
    return {"requested_url": url, "final_url": final_url,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(), "local_path": destination.name}


def run(project_root):
    root = Path(project_root).resolve()
    if not (root / "configs/data_sources.json").is_file():
        raise ValueError("Sync configs/data_sources.json first")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = root / "reports/phase2" / ("p2_02-" + run_id)
    folder.mkdir(parents=True, exist_ok=False)
    documents = folder / "documents"
    documents.mkdir()
    records, failures = [], []

    def fetch(name, url):
        try:
            result = capture(url, documents / name)
            result["local_path"] = "documents/" + name
            records.append(result)
            return documents / name
        except Exception as exc:
            failures.append({"document": name, "url": url, "error": str(exc)})
            return None

    zenodo = fetch("phonemedf_release.json", "https://zenodo.org/api/records/19522979")
    fetch("phonemedf_documentation.pdf", "https://zenodo.org/records/19522979/files/PhonemeDF_Documentation.pdf?download=1")
    commit = fetch("phonemedf_commit.json", "https://api.github.com/repos/Vamshi-Nallaguntla/PhonemeDF/commits/main")
    github_sha = None
    if commit:
        try:
            github_sha = json.loads(commit.read_text(encoding="utf-8"))["sha"]
            if not re.fullmatch(r"[0-9a-f]{40}", github_sha):
                raise ValueError("Invalid repository commit identifier")
            fetch("phonemedf_README.md", f"https://raw.githubusercontent.com/Vamshi-Nallaguntla/PhonemeDF/{github_sha}/README.md")
        except (KeyError, ValueError) as exc:
            failures.append({"document": "phonemedf_commit.json", "error": str(exc)})
    fetch("mlaad_tiny_README.md", f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{HF_REVISION}/README.md")
    fetch("mlaad_tiny_original_LICENSE", f"https://huggingface.co/datasets/mueller91/MLAAD-tiny/resolve/{HF_REVISION}/original/LICENSE")
    fetch("mlaad_tiny_repository.json", f"https://huggingface.co/api/datasets/mueller91/MLAAD-tiny/revision/{HF_REVISION}")
    publisher = None
    if zenodo:
        try:
            value = json.loads(zenodo.read_text(encoding="utf-8"))
            publisher = {"id": value["id"], "metadata": value["metadata"], "files": value["files"]}
        except (KeyError, ValueError) as exc:
            failures.append({"document": "phonemedf_release.json", "error": str(exc)})
    registry = root / "configs/data_sources.json"
    inventory = {"schema_version": 1, "task": "P2-02", "run_id": run_id,
                 "status": "documents_captured_for_review" if not failures else "needs_attention",
                 "scope": "Publisher metadata/documents only; no audio acquisition or admission",
                 "documents": records, "failures": failures, "phonemedf_publisher_metadata": publisher,
                 "phonemedf_repository_commit": github_sha, "mlaad_tiny_metadata_revision": HF_REVISION,
                 "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                 "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "local_publisher_revision_verified": False,
                 "permissions_review": "pending inspection of captured terms and inherited conditions",
                 "external_audio_availability": "not_downloaded_user_confirmed"}
    (folder / "source_inventory.json").write_bytes((json.dumps(inventory, indent=2) + "\n").encode())
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(folder.rglob("*")) if p.is_file()}
    (folder / "artifact_manifest.json").write_bytes((json.dumps({"schema_version": 1, "algorithm": "sha256", "files": files}, indent=2) + "\n").encode())
    print(json.dumps({"status": inventory["status"], "evidence_dir": str(folder), "failures": failures}, indent=2))
    return inventory, folder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    report, _ = run(args.project_root)
    raise SystemExit(0 if not report["failures"] else 4)
