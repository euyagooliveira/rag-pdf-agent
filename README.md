# Agente RAG para Documentos PDF

Sistema de Retrieval-Augmented Generation (RAG) que permite fazer upload de qualquer PDF e conversar com o conteúdo do documento em linguagem natural.

---

## Sobre o projeto

A maioria das aplicações de LLM enfrenta um problema comum: o modelo não conhece informações privadas ou específicas de um domínio, que não fizeram parte do seu treinamento. O RAG resolve isso buscando, em tempo real, os trechos mais relevantes de um documento para cada pergunta, e fornecendo esse contexto ao modelo antes de gerar a resposta.

Isso reduz drasticamente a alucinação: o modelo deixa de "inventar" respostas e passa a se basear no conteúdo real do documento enviado, admitindo claramente quando a informação não está presente.

---

## Arquitetura

```
        ┌──────────────────┐
        │   Upload do PDF    │
        └─────────┬──────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ Extração de texto  │
        └─────────┬──────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Divisão em chunks │
        │  (com sobreposição)│
        └─────────┬──────────┘
                   │
                   ▼
        ┌──────────────────┐
        │ Geração de        │
        │ embeddings local   │
        └─────────┬──────────┘
                   │
   ┌───────────────┴────────────────┐
   │                                 │
   ▼                                 │
┌──────────────────┐                │
│ Pergunta do       │                │
│ usuário            │                │
└─────────┬──────────┘                │
          │                           │
          ▼                           │
┌──────────────────┐                  │
│ Embedding da      │                  │
│ pergunta           │                  │
└─────────┬──────────┘                  │
          │                             │
          ▼                             │
┌──────────────────┐ ◄──────────────────┘
│ Busca por          │
│ similaridade        │
│ (cosseno)           │
└─────────┬──────────┘
          │
          ▼
┌──────────────────┐
│ LLM responde com  │
│ base nos trechos   │
│ mais relevantes    │
└──────────────────┘
```

---

## Exemplo de uso

**Documento:** um manual técnico em PDF

**Pergunta:** "Qual é o prazo de garantia do produto?"

**Resposta:** o agente busca os trechos do documento semanticamente mais próximos da pergunta, envia esses trechos ao LLM junto com a pergunta, e retorna a resposta com base apenas no que está escrito no manual.

**Pergunta fora do escopo:** "Qual a previsão do tempo para amanhã?"

**Resposta:** o agente reconhece que essa informação não está no documento e informa isso claramente, em vez de inventar uma resposta.

---

## Tecnologias

- Groq API — inferência com LLaMA 3.3 70B para geração da resposta final
- sentence-transformers — geração de embeddings local, sem custo de API (modelo all-MiniLM-L6-v2)
- pypdf — extração de texto de arquivos PDF
- NumPy — cálculo de similaridade de cosseno entre embeddings
- Streamlit — interface de upload e chat

---

## Como rodar localmente

**1. Clone o repositório**
```bash
git clone https://github.com/euyagooliveira/rag-pdf-agent.git
cd rag-pdf-agent
```

**2. Instale as dependências**
```bash
pip3 install -r requirements.txt
```

**3. Gere uma API Key gratuita no Groq**

Acesse [console.groq.com](https://console.groq.com), crie uma conta e gere uma API Key em API Keys → Create API Key.

**4. Execute a aplicação**
```bash
python3 -m streamlit run app.py
```

**5. Use a aplicação**

Cole sua Groq API Key na barra lateral, faça upload de um PDF e comece a fazer perguntas sobre o conteúdo do documento.

---

## Estrutura do projeto

```
rag-pdf-agent/
├── app.py              # Interface Streamlit (upload, chat e exibição de trechos usados)
├── rag.py               # Lógica de extração, chunking, embeddings e busca semântica
└── requirements.txt     # Dependências do projeto
```

---

## Observações

A busca semântica é feita por similaridade de cosseno calculada diretamente em memória com NumPy, sem uso de um banco de vetores dedicado. Essa abordagem é adequada para o processamento de um único documento por sessão, mas não escala para uma base com muitos documentos persistidos simultaneamente.

---

## Autor

**Yago Oliveira**
Manager & Staff AI Engineer | Estatístico 