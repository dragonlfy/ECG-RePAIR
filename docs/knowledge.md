# Knowledge sources and retrieval

No textbook PDFs or extracted textbook corpus are bundled. The source project
used the following educational resources, recorded as CC BY 4.0 in its source
manifest:

| Resource | Source |
| --- | --- |
| *Nursing Advanced Skills*, Chapter 7: Interpret Basic ECG; edited by Kim Ernstmeyer and Elizabeth Christman, Open RN (2023) | https://www.ncbi.nlm.nih.gov/books/NBK594493/ |
| *An EKG Interpretation Primer*, Jacqueline Christianson and the Nurses International team (2019) | https://open.umn.edu/opentextbooks/textbooks/an-ekg-interpretation-primer |

Obtain your own copies from the publishers and check the applicable license and
any third-party exceptions before redistribution. Preserve title, authors,
source URL, page, and license in each extracted passage.

`ClinicalKnowledgeIndex.from_jsonl` accepts one object per passage:

```json
{
  "chunk_id": "your-source:p12:c1",
  "text": "Your licensed source passage.",
  "content_sha256": "SHA-256 of the passage text",
  "metadata": {
    "source_id": "your-source",
    "title": "Source title",
    "page": 12,
    "license": "Source license",
    "catalog_url": "https://example.org/source",
    "authority_tier": "educational_source"
  }
}
```

Use `python examples/index_pdf.py --help` to index a locally obtained PDF. This
utility requires `python -m pip install -e '.[rag]'`. It retains the PDF page number;
check it against printed page numbering if they differ. Passage extraction does
not certify the source text or convert it automatically into executable rules.

The library supports `retrieve(query, k=3)` and optional `source_ids` filtering.
Any extracted diagnostic threshold must be reviewed against the intended clinical
criterion before changing the verifier. Source retrieval and patient measurement
remain distinct inputs.
