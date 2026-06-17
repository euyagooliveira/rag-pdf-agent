# app.py
# Interface Streamlit para o agente RAG de PDFs

import streamlit as st
from groq import Groq
from rag import extrair_texto, criar_chunks, gerar_embeddings, buscar_chunks_relevantes

# Configuração da página
st.set_page_config(
    page_title="Agente RAG - Converse com seu PDF",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Converse com seu PDF")
st.caption("Faça upload de um documento e tire dúvidas sobre o conteúdo dele.")

# API Key na barra lateral
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_..."
)

if not api_key:
    st.warning("Insira sua Groq API Key na barra lateral para começar.")
    st.stop()

cliente = Groq(api_key=api_key)

# Upload do PDF
arquivo_pdf = st.sidebar.file_uploader("Envie um PDF", type=["pdf"])

if not arquivo_pdf:
    st.info("Envie um PDF na barra lateral para começar a conversa.")
    st.stop()

# Processa o PDF apenas uma vez por upload (usando cache de sessão)
if "chunks" not in st.session_state or st.session_state.get("arquivo_atual") != arquivo_pdf.name:
    with st.spinner("Processando o documento..."):
        texto = extrair_texto(arquivo_pdf)
        chunks = criar_chunks(texto)
        embeddings = gerar_embeddings(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.arquivo_atual = arquivo_pdf.name
        st.session_state.mensagens = []

    st.sidebar.success(f"Documento processado: {len(chunks)} trechos indexados.")

# Histórico de mensagens
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
pergunta = st.chat_input("Pergunte algo sobre o documento...")

if pergunta:
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.mensagens.append({"role": "user", "content": pergunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando no documento..."):
            # Busca os trechos mais relevantes para a pergunta
            chunks_relevantes = buscar_chunks_relevantes(
                pergunta,
                st.session_state.chunks,
                st.session_state.embeddings
            )

            contexto = "\n\n".join(chunks_relevantes)

            resposta = cliente.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """Você responde perguntas com base APENAS no contexto fornecido,
                        extraído de um documento PDF. Se a resposta não estiver no contexto,
                        diga claramente que não encontrou essa informação no documento.
                        Responda em português de forma clara e objetiva."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Contexto extraído do documento:
                        {contexto}

                        Pergunta: {pergunta}
                        """
                    }
                ]
            )

            resposta_texto = resposta.choices[0].message.content.strip()
            st.markdown(resposta_texto)

            with st.expander("📋 Trechos do documento usados"):
                for i, chunk in enumerate(chunks_relevantes, 1):
                    st.markdown(f"**Trecho {i}:**")
                    st.text(chunk[:300] + "...")

    st.session_state.mensagens.append({"role": "assistant", "content": resposta_texto})