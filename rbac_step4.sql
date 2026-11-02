-- Add the 'registered' label (for signed-up customers) to the allowed labels
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_visibility_check;
ALTER TABLE documents ADD CONSTRAINT documents_visibility_check
    CHECK (visibility IN ('public', 'registered', 'internal', 'department', 'admin_only'));