"""molpredict backend: draw a molecule, get predictions from published models."""


def main() -> None:
    import uvicorn

    uvicorn.run("molpredict_backend.api:app", host="127.0.0.1", port=8000)
