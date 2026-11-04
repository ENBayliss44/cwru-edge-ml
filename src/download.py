"""Download the CWRU 12 kHz drive-end recordings listed in data/manifest.csv."""
import csv
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://engineering.case.edu/sites/default/files/{}.mat"


def main():
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data" / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        dst = out / f"{r['file_id']}.mat"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"fetching {dst.name}")
        urllib.request.urlretrieve(URL.format(r["file_id"]), dst)
    print(f"{len(rows)} files in {out}")


if __name__ == "__main__":
    main()
