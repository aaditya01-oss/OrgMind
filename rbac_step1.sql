BEGIN;

-- Who belongs to which department (empty for now: an admin sets it when inviting)
ALTER TABLE users   ADD COLUMN IF NOT EXISTS department VARCHAR(100);
ALTER TABLE invites ADD COLUMN IF NOT EXISTS department TEXT;

-- One row per document, holding WHO may see it
CREATE TABLE IF NOT EXISTS documents (
    org_id      VARCHAR(100) NOT NULL,
    doc_name    VARCHAR(255) NOT NULL,
    visibility  VARCHAR(20)  NOT NULL DEFAULT 'internal'
                CHECK (visibility IN ('public', 'internal', 'department', 'admin_only')),
    department  VARCHAR(100),
    uploaded_by VARCHAR(255),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, doc_name),
    CHECK (visibility <> 'department' OR department IS NOT NULL)
);

-- Every document you already have becomes 'internal' (everyone in the org), the same as today
INSERT INTO documents (org_id, doc_name)
SELECT DISTINCT org_id, doc_name FROM document_chunks
ON CONFLICT DO NOTHING;

COMMIT;