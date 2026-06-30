from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HGE/tkbc ICEWS14 pickles without installing the package.")
    parser.add_argument("--hge-root", default=Path(os.environ.get("HGE_ROOT", "../HGE-main")), type=Path)
    parser.add_argument("--dataset", default="ICEWS14")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tkbc_root = args.hge_root / "tkbc"
    data_root = tkbc_root / "data"
    out_dir = data_root / args.dataset
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists():
        print(f"exists: {out_dir}")
        return

    sys.path.insert(0, str(tkbc_root))
    import pkg_resources  # type: ignore

    original = pkg_resources.resource_filename

    def resource_filename(package: str, resource: str) -> str:
        if package == "tkbc" and resource == "data/":
            return str(data_root) + "/"
        return original(package, resource)

    pkg_resources.resource_filename = resource_filename  # type: ignore
    import process_icews  # type: ignore

    process_icews.DATA_PATH = str(data_root) + "/"
    os.chdir(tkbc_root)
    process_icews.prepare_dataset(str(tkbc_root / "src_data" / args.dataset), args.dataset)
    print(f"prepared: {out_dir}")


if __name__ == "__main__":
    main()
