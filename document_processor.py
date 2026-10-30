from contextlib import closing
import os
import unicodedata
import psycopg2
from embeddings import get_embedding
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


class DocumentError(ValueError):
    """A problem with the uploaded document whose message is safe to show to the user."""

def clean_doc_name(raw: str, max_len: int = 120) -> str:
    """Make an uploaded file name safe and tidy for display and storage."""
    name = unicodedata.normalize("NFC", raw or "")
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")  # control characters
    name = name.replace("\\", "/").rsplit("/", 1)[-1]                         # file name only, no paths
    name = " ".join(name.split())                                             # tidy spaces
    i = 0
    while i < len(name) and not name[i].isalnum():                            # drop leading junk like "### "
        i += 1
    name = name[i:]
    stem, ext = os.path.splitext(name)
    if len(name) > max_len:
        stem = stem[: max_len - len(ext)].rstrip()
        name = stem + ext
    if not name or not stem:
        raise DocumentError("This file name is not valid. Please rename the file and try again.")
    return name

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def chunk_text(text, chunk_size=120, overlap=25):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


def _pdf_text(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(path):
    """Paragraphs AND tables, in document order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(path)
    parts = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(p for p in parts if p.strip())


def _txt_text(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


EXTRACTORS = {"pdf": _pdf_text, "docx": _docx_text, "txt": _txt_text}


def extract_text(file_path):
    extension = file_path.lower().rsplit(".", 1)[-1]
    extractor = EXTRACTORS.get(extension)
    if extractor is None:
        raise DocumentError(f"Unsupported format: .{extension}")
    return extractor(file_path)


def process_document(file_path, org_id, doc_name):
    """Extract, chunk, embed and store a document. Returns the number of chunks.

    Uploading a document with the same name again REPLACES the old version
    (inside one transaction, so a failure leaves the old version untouched).
    """
    text = extract_text(file_path)
    chunks = chunk_text(text)

    if not chunks:
        raise DocumentError(
            "No readable text was found in this file. If it is a scan or a photo, "
            "use 'Scan Document' instead."
        )

    # The slow part (embedding) happens BEFORE we touch the database.
    rows = [(org_id, doc_name, i, chunk, get_embedding(chunk)) for i, chunk in enumerate(chunks)]

    with closing(get_db_connection()) as conn:
        with conn:  # one transaction: commit on success, roll back on any error
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE org_id = %s AND doc_name = %s",
                    (org_id, doc_name),
                )
                cur.executemany(
                    """
                    INSERT INTO document_chunks
                    (org_id, doc_name, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    print(f"Processed {len(chunks)} chunks from {doc_name}")
    return len(chunks)


if __name__ == "__main__":
    pdf_path = input("Enter file path: ")
    org_id = input("Enter organization ID: ")
    doc_name = input("Enter document name: ")
    process_document(pdf_path, org_id, doc_name)