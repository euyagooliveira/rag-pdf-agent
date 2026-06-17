# rag.py
# Lógica central do RAG: extração de texto, chunking, embeddings e busca

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Carrega o modelo de embeddings uma única vez
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')


def extrair_texto(arquivo_pdf) -> str:
    """
    Recebe um arquivo PDF e retorna todo o texto extraído.
    """
    leitor = PdfReader(arquivo_pdf)
    texto_completo = ""

    for pagina in leitor.pages:
        texto_completo += pagina.extract_text() + "\n"

    return texto_completo


def criar_chunks(texto: str, tamanho_chunk: int = 500, sobreposicao: int = 50) -> list:
    """
    Divide o texto em pedaços menores (chunks), com uma sobreposição entre eles
    para não perder contexto nas bordas de cada chunk.
    """
    palavras = texto.split()
    chunks = []

    inicio = 0
    while inicio < len(palavras):
        fim = inicio + tamanho_chunk
        chunk = " ".join(palavras[inicio:fim])
        chunks.append(chunk)
        inicio += tamanho_chunk - sobreposicao

    return chunks


def gerar_embeddings(chunks: list) -> np.ndarray:
    """
    Transforma cada chunk de texto em um vetor numérico (embedding).
    """
    return modelo_embeddings.encode(chunks)


def buscar_chunks_relevantes(pergunta: str, chunks: list, embeddings: np.ndarray, top_k: int = 3) -> list:
    """
    Busca os chunks mais relevantes para a pergunta, usando similaridade de cosseno.
    """
    embedding_pergunta = modelo_embeddings.encode([pergunta])

    # Similaridade de cosseno entre a pergunta e cada chunk
    similaridades = np.dot(embeddings, embedding_pergunta.T).flatten()
    similaridades = similaridades / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embedding_pergunta)
    )

    # Pega os índices dos top_k chunks mais similares
    indices_top = np.argsort(similaridades)[-top_k:][::-1]

    return [chunks[i] for i in indices_top]