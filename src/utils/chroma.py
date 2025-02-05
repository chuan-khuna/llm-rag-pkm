import chromadb
from chromadb import Settings


def get_chroma_client(host: str, port: int, auth_token: str):
    chroma_client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=auth_token,
        ),
    )
    return chroma_client
