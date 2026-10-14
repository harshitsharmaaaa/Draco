import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.chat import read_user, upload_user
from app.core.queue import ingestion_queue
from app.db.registry import (
    FAILED_STATUS,
    get_document,
    get_pages,
    register_upload,
    update_status,
    set_job_id,
)
from app.ingestion.worker import process_document

router = APIRouter()

UPLOAD_DIR = (
    Path(__file__).resolve().parents[1]
    / "storage"
    / "uploads"
)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/documents", status_code=202, dependencies=[Depends(upload_user)])
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid4())

    # Generate the storage filename ourselves.
    # Do not trust the uploaded filename as a filesystem path.
    safe_filename = Path(file.filename).name
    if not safe_filename or not safe_filename.strip() or safe_filename in (".", ".."):
        raise HTTPException(400, "Filename is required")
    if len(safe_filename) > 100:
        suffix = Path(safe_filename).suffix[:16]
        safe_filename = safe_filename[: 100 - len(suffix)] + suffix
    if Path(safe_filename).suffix.lower() != ".pdf":
        # The worker only converts PDFs; failing here beats queueing a job that is certain to fail.
        raise HTTPException(415, "Only PDF files are supported")
    file_path = UPLOAD_DIR / f"{document_id}_{safe_filename}"

    size = 0
    digest = hashlib.sha256()
    registered = False

    try:
        with file_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                if size == 0 and not chunk.lstrip(b"\x00 \t\r\n")[:5] == b"%PDF-":
                    raise HTTPException(415, "File content is not a PDF")
                size += len(chunk)
                digest.update(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        413,
                        "File exceeds the 100 MB limit",
                    )

                destination.write(chunk)

        existing = register_upload(
            document_id=document_id,
            filename=safe_filename,
            file_path=str(file_path.resolve()),
            content_hash=digest.hexdigest(),
        )

        if existing is not None:
            # Identical content is already registered: keep one copy and point
            # the caller at it instead of ingesting it twice.
            file_path.unlink(missing_ok=True)
            return JSONResponse(
                status_code=200,
                content={
                    "document_id": existing["document_id"],
                    "job_id": existing["job_id"],
                    "filename": existing["filename"],
                    "status": existing["status"],
                    "duplicate": True,
                    "message": "Identical file already ingested; no new job queued",
                },
            )

        registered = True

        update_status(document_id, "queued")

        job = ingestion_queue.enqueue(
            process_document,
            document_id,
            str(file_path.resolve()),
            job_timeout=3600,
            result_ttl=86400,
        )

        set_job_id(document_id, job.id)

        return {
            "document_id": document_id,
            "job_id": job.id,
            "filename": safe_filename,
            "status": "queued",
            "duplicate": False,
            "message": "File accepted for ingestion",
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        file_path.unlink(missing_ok=True)
        if registered:
            # Otherwise this row would stay "queued" forever and make every
            # re-upload of the same file look like a duplicate of it.
            update_status(document_id, FAILED_STATUS, error="Could not queue the file")
        raise HTTPException(
            503,
            "Could not queue the file. Please retry.",
        ) from exc

    finally:
        await file.close()

@router.get("/documents/{document_id}", dependencies=[Depends(read_user)])
def document_status(document_id: str):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(404, "Document not found")

    return document


@router.get("/documents/{document_id}/pages", dependencies=[Depends(read_user)])
def document_pages(document_id: str):
    if get_document(document_id) is None:
        raise HTTPException(404, "Document not found")

    return get_pages(document_id)
