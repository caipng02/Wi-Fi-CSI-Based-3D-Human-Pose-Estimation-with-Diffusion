"""Restore the exact audited P3-S1 split manifest used by the Atlas experiment."""

import gzip
import hashlib
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "6fdbb55650dc7c151d4b50241ab82186721b2b858ba7b7ae6ee1d457d99b10a0"


def main():
    target = ROOT / "data/p3-s1.json"
    data = gzip.decompress((ROOT / "data/p3-s1.json.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("The compressed split manifest checksum does not match.")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"A different manifest already exists at {target}.")
    else:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            stream.write(data)
            temporary = Path(stream.name)
        temporary.replace(target)
    print(f"Ready: {target}")
    print(f"SHA256: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
