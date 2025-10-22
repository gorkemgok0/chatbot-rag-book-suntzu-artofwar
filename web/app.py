"""
Sun Tzu RAG Chatbot - Streamlit Web Arayüzü
Sun Tzu'nun Savaş Sanatı öğretilerine dayalı akıllı sohbet asistanı
"""

import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import time

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sun Tzu Chatbot", 
    page_icon="⚔️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_rag_components():
    """
    RAG bileşenlerini yükler ve cache'ler
    """
    # Ortam değişkenlerini yükle
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # API key kontrolü
    if not api_key:
        st.error("GOOGLE_API_KEY bulunamadı! Lütfen .env dosyasında API key'inizi ayarlayın.")
        return None, None, None, None
    
    if not api_key:
        st.error("GOOGLE_API_KEY bulunamadı! .env dosyasını kontrol edin.")
        return None, None, None, None
    
    # Gemini yapılandırması
    genai.configure(api_key=api_key)
    
    # Model kontrolü ve seçimi
    try:
        models = genai.list_models()
        available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # En uygun modeli seç
        if "models/gemini-1.5-flash-latest" in available_models:
            MODEL_NAME = "gemini-1.5-flash-latest"
        elif "models/gemini-1.5-flash" in available_models:
            MODEL_NAME = "gemini-1.5-flash"
        elif "models/gemini-flash-latest" in available_models:
            MODEL_NAME = "gemini-flash-latest"
        else:
            MODEL_NAME = available_models[0] if available_models else "gemini-1.5-flash-latest"
            
        print(f"Seçilen model: {MODEL_NAME}")
    except Exception as e:
        print(f"Model listesi alınamadı: {e}")
        MODEL_NAME = "gemini-1.5-flash-latest"
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # ChromaDB bağlantısı (kalıcı veritabanı)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = chroma_client.get_collection("sun_tzu_collection")
    except:
        st.warning("ChromaDB koleksiyonu bulunamadı, yeniden oluşturuluyor...")
        
        # Embedding'i yeniden oluştur
        try:
            with open("sun_tzu.txt", "r", encoding="utf-8") as f:
                texts = [t.strip() for t in f.readlines() if t.strip() and len(t.strip()) > 10]
            
            collection = chroma_client.create_collection(name="sun_tzu_collection")
            
            # Embedding'leri oluştur
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(texts):
                emb = embedder.encode(text).tolist()
                collection.add(
                    ids=[str(i)],
                    embeddings=[emb],
                    documents=[text]
                )
                
                # Progress bar güncelle
                progress = (i + 1) / len(texts)
                progress_bar.progress(progress)
                status_text.text(f"Embedding oluşturuluyor... {i+1}/{len(texts)}")
            
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Embedding oluşturulamadı: {str(e)}")
            return None, None, None, None
    
    # Gemini modeli
    model = genai.GenerativeModel(MODEL_NAME)
    
    return model, embedder, collection, MODEL_NAME

def get_response(model, embedder, collection, query):
    """
    RAG pipeline ile cevap üretir
    """
    try:
        # Retriever kısmı
        query_emb = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=3)
        context = "\n\n".join(results["documents"][0])
        
        # Prompt oluştur
        prompt = f"""
Sen Sun Tzu'nun Savaş Sanatı konusunda uzman bir asistan olarak görev yapıyorsun. 
Aşağıdaki bağlamda (context) Sun Tzu'nun öğretileri yer almaktadır.

Bağlam:
{context}

Soru: {query}

Lütfen bu bağlamdaki bilgilere dayanarak kısa ve anlaşılır bir yanıt ver. 
Sun Tzu'nun öğretilerini kullanarak pratik öneriler sun.
Eğer bağlamda doğrudan bilgi yoksa, benzer konulardaki öğretileri kullanarak yanıt ver.
"""
        
        # Gemini API çağrısı
        response = model.generate_content(prompt)
        return response.text, context
        
    except Exception as e:
        return f"Üzgünüm, bir hata oluştu: {str(e)}", ""

def main():
    """
    Ana uygulama fonksiyonu
    """
    # Başlık ve açıklama
    st.title("⚔️ Sun Tzu - Savaş Sanatı Chatbot")
    st.markdown("---")
    st.markdown("**Sun Tzu'nun öğretilerine dayalı akıllı bir sohbet asistanı**")
    st.markdown("Savaş stratejisi, liderlik, taktik ve stratejik düşünce konularında sorularınızı sorun!")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Hakkında")
        st.markdown("""
        Bu chatbot, Sun Tzu'nun **Savaş Sanatı** eserinin Türkçe çevirisini 
        kullanarak RAG (Retrieval-Augmented Generation) teknolojisi ile çalışır.
        
        **Teknoloji Stack:**
        - 🤖 Gemini 1.5 Flash (LLM)
        - 🧠 Sentence Transformers (Embedding)
        - 🗄️ ChromaDB (Vector Database)
        - 🚀 Streamlit (Web Arayüzü)
        """)
        
        st.header("💡 Örnek Sorular")
        example_questions = [
            "Savaşta strateji nasıl belirlenir?",
            "Düşman nasıl yenilir?",
            "Ordunun moralini nasıl yükseltirsin?",
            "Kaynakları nasıl yönetirsin?",
            "Liderlik nasıl olmalıdır?",
            "Ne zaman savaşmamak gerekir?"
        ]
        
        for question in example_questions:
            if st.button(f"❓ {question}", key=f"example_{question}"):
                st.session_state.user_input = question
                st.rerun()
    
    # RAG bileşenlerini yükle
    with st.spinner("RAG sistemi yükleniyor..."):
        model, embedder, collection, model_name = load_rag_components()
    
    if not all([model, embedder, collection]):
        st.error("RAG sistemi yüklenemedi! Lütfen konsol çıktısını kontrol edin.")
        return
    
    # Kullanıcı girdisi
    st.markdown("### 💬 Soru Sor")
    
    # Örnek soru butonundan gelen değer
    default_value = st.session_state.get('user_input', '')
    if default_value:
        st.session_state.user_input = ''  # Reset
    
    query = st.text_input(
        "Sun Tzu'nun öğretileri hakkında bir soru sorun:",
        value=default_value,
        placeholder="Örn: Savaşta strateji nasıl belirlenir?",
        help="Savaş stratejisi, liderlik, taktik veya stratejik düşünce konularında soru sorabilirsiniz."
    )
    
    if query:
        with st.spinner("Düşünüyorum..."):
            # RAG pipeline ile cevap üret
            response, context = get_response(model, embedder, collection, query)
        
        # Yanıtı göster
        st.markdown("### 🎯 Cevap")
        st.markdown(response)
        
        # Kaynakları göster (genişletilebilir)
        if context:
            with st.expander("📚 Kullanılan Kaynak Metinler"):
                st.text_area("Bağlam:", context, height=200, disabled=True)
            
            # Performans bilgisi
            st.markdown("---")
            st.markdown("**💡 Bu cevap Sun Tzu'nun Savaş Sanatı eserinden alınan metinler temel alınarak üretilmiştir.**")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>Sun Tzu RAG Chatbot | RAG + Gemini + Streamlit</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()