import pytest
from document_loader import build_multimodal_content

def test_build_multimodal_content_gcs():
    contents = build_multimodal_content("gs://my-bucket/sample.pdf", "Extract data")
    assert len(contents) >= 2
    assert contents[0]["type"] == "media"
    assert contents[0]["file_uri"] == "gs://my-bucket/sample.pdf"

def test_build_multimodal_content_raw_text():
    contents = build_multimodal_content("This is raw text from document.", "Analyze:")
    assert len(contents) == 1
    assert contents[0]["type"] == "text"
    assert "Analyze:" in contents[0]["text"]
    assert "This is raw text from document." in contents[0]["text"]

def test_build_multimodal_content_local_pdf(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.5 test binary content")
    contents = build_multimodal_content(str(pdf_file), "Extract:")
    assert len(contents) == 2
    assert contents[0]["type"] == "media"
    assert contents[0]["mime_type"] == "application/pdf"
    assert contents[0]["data"] is not None
    assert contents[1]["text"] == "Extract:"

def test_build_multimodal_content_local_text_file(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Local text content")
    contents = build_multimodal_content(str(txt_file), "Extract:")
    assert len(contents) == 1
    assert contents[0]["type"] == "text"
    assert "Extract:" in contents[0]["text"]
    assert "Local text content" in contents[0]["text"]

