import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

class PDFQAChatbot:
    def __init__(self):
        # Carrega o pipeline de question answering usando um modelo para português
        self.qa_pipeline = pipeline(
            "question-answering", 
            model="pierreguillou/bert-base-portuguese-cased-squad-v1",
            tokenizer="pierreguillou/bert-base-portuguese-cased-squad-v1"
        )
        self.pdf_text = ""
    
    def extract_pdf_text(self, pdf_file):
        """
        Extrai o texto do PDF concatenando as páginas.
        """
        try:
            pdf_reader = PdfReader(pdf_file)
            pages = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text is not None:
                    pages.append(text.strip())
            self.pdf_text = " ".join(pages)
            return True
        except Exception as e:
            st.error("Erro ao extrair texto do PDF: " + str(e))
            return False
    
    def find_relevant_context(self, query, window=700):
        """
        Localiza um trecho relevante baseado na primeira ocorrência de alguma palavra da query.
        Retorna uma janela de 'window' caracteres ao redor dessa ocorrência.
        """
        query_lower = query.lower()
        text_lower = self.pdf_text.lower()
        # Procura a posição da primeira palavra da query
        first_word = query_lower.split()[0]
        pos = text_lower.find(first_word)
        if pos == -1:
            pos = 0
        start = max(pos - window // 2, 0)
        end = min(pos + window // 2, len(self.pdf_text))
        return self.pdf_text[start:end]
    
    def generate_response(self, query):
        """
        Gera a resposta usando o pipeline de QA com base no contexto extraído.
        """
        context = self.find_relevant_context(query)
        if not context:
            return "Não foi possível encontrar um contexto relevante no documento."
        
        try:
            result = self.qa_pipeline(question=query, context=context)
            answer = result.get('answer', '').strip()
            response = (
                f"De acordo com o documento, {answer} \n\n"
                f"(Contexto utilizado: \"{context[:300]}...\")"
            )
            return response
        except Exception as e:
            return "Erro ao gerar resposta: " + str(e)

def main():
    st.set_page_config(page_title="Assistente de PDF", page_icon="📚")
    st.title("Assistente de Documentos PDF")
    st.subheader("Obtenha respostas fundamentadas no texto do documento")
    
    chatbot = PDFQAChatbot()
    
    uploaded_file = st.file_uploader("Carregue seu PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extraindo texto do documento..."):
            if chatbot.extract_pdf_text(uploaded_file):
                st.success("Documento carregado com sucesso!")
                with st.expander("Prévia do texto extraído"):
                    st.write(chatbot.pdf_text[:500] + "...")
    
    query = st.text_input("Faça sua pergunta sobre o documento")
    
    if st.button("Obter Resposta") and query:
        if not chatbot.pdf_text:
            st.warning("Por favor, carregue um PDF primeiro.")
        else:
            with st.spinner("Gerando resposta..."):
                response = chatbot.generate_response(query)
                st.markdown("### Resposta")
                st.write(response)

if __name__ == "__main__":
    main()
