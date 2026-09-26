"""Index a user-supplied PDF with source metadata; no download is performed."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int)
    args = parser.parse_args()
    from pypdf import PdfReader
    reader = PdfReader(args.pdf)
    last = args.last_page or len(reader.pages)
    if not 1 <= args.first_page <= last <= len(reader.pages):
        parser.error("page range is outside the PDF")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for number in range(args.first_page, last + 1):
            words = (reader.pages[number - 1].extract_text() or "").split()
            for start in range(0, len(words), 180):
                text = " ".join(words[start:start + 220])
                row = {
                    "chunk_id": f"{args.source_id}:p{number}:c{start // 180}",
                    "text": text,
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "metadata": {
                        "source_id": args.source_id, "title": args.title,
                        "page": number, "license": args.license,
                        "catalog_url": args.url, "authority_tier": "educational_source",
                    },
                }
                output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
