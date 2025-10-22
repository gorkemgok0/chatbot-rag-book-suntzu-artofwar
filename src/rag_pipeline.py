"""
RAG Chatbot - Retriever + Gemini Generator Pipeline
Sun Tzu metinlerinden anlamsal arama yapıp Gemini ile cevap üretme
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

def setup_rag_pipeline():
    """
    RAG pipeline'ını kurar ve yapılandırır
    """
    print("RAG Pipeline kurulumu basliyor...")
    
    # 1. Ortam değişkenlerini yükle
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # API key kontrolü
    if not api_key:
        print("HATA: GOOGLE_API_KEY bulunamadi! .env dosyasini kontrol edin.")
        return None, None, None
    
    if not api_key:
        print("HATA: GOOGLE_API_KEY bulunamadi! .env dosyasini kontrol edin.")
        return None, None, None
    
    genai.configure(api_key=api_key)
    print("Google API yapilandirildi")
    
    # 2. Model seçimi (ücretsiz, hızlı, az token harcayan)
    MODEL_NAME = "gemini-1.5-flash"
    print(f"Model secildi: {MODEL_NAME}")
    
    # 3. Embedding modeli ve Chroma retriever ayarı
    print("Embedding modeli yukleniyor...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("ChromaDB baglantisi kuruluyor...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = chroma_client.get_collection("sun_tzu_collection")
        print("ChromaDB koleksiyonu bulundu")
    except:
        print("ChromaDB koleksiyonu bulunamadi, yeniden olusturuluyor...")
        
        # Embedding'i yeniden oluştur
        with open("data/sun_tzu.txt", "r", encoding="utf-8") as f:
            texts = [t.strip() for t in f.readlines() if t.strip() and len(t.strip()) > 10]
        
        collection = chroma_client.create_collection(name="sun_tzu_collection")
        print(f"{len(texts)} metin embedding'e donusturuluyor...")
        
        for i, text in enumerate(texts):
            emb = embedder.encode(text).tolist()
            collection.add(
                ids=[str(i)],
                embeddings=[emb],
                documents=[text]
            )
        
        print("Embedding tamamlandi!")
    
    # 4. Mevcut modelleri listele
    print("Mevcut modeller kontrol ediliyor...")
    try:
        models = genai.list_models()
        print("Mevcut modeller:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
    except Exception as e:
        print(f"Model listesi alinamadi: {e}")
    
    # 5. Gemini modeli
    model = genai.GenerativeModel(MODEL_NAME)
    print("Gemini modeli hazir")
    
    print("RAG Pipeline basariyla kuruldu!")
    return model, embedder, collection

def get_response(model, embedder, collection, query):
    """
    Kullanıcı sorgusuna RAG pipeline ile cevap üretir
    """
    print(f"\nSorgu isleniyor: '{query}'")
    
    # 1. En alakalı dokümanları getir
    print("En alakali metinler araniyor...")
    query_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    context = "\n\n".join(results["documents"][0])
    
    print(f"Bulunan {len(results['documents'][0])} alakali metin parcası")
    print(f"Context (ilk 200 karakter): {context[:200]}...")
    
    # 2. Prompt oluştur
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
    
    # 3. Gemini API çağrısı
    print("Gemini ile cevap uretiliyor...")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API hatasi: {e}")
        return "Uzgunum, bir hata olustu. Lutfen tekrar deneyin."

def interactive_chat():
    """
    Etkileşimli chat modu
    """
    model, embedder, collection = setup_rag_pipeline()
    
    if not all([model, embedder, collection]):
        return
    
    print("\n" + "="*50)
    print("Sun Tzu RAG Chatbot")
    print("Cikmak icin 'cikis' yazin")
    print("="*50)
    
    while True:
        query = input("\nSoru: ").strip()
        
        if query.lower() in ['cikis', 'exit', 'quit', 'q']:
            print("Gorusmek uzere!")
            break
        
        if not query:
            continue
        
        response = get_response(model, embedder, collection, query)
        print(f"\nCevap: {response}")

def single_query():
    """
    Tek sorgu modu
    """
    model, embedder, collection = setup_rag_pipeline()
    
    if not all([model, embedder, collection]):
        return
    
    query = input("Soru: ")
    response = get_response(model, embedder, collection, query)
    print(f"\nCevap: {response}")

def main():
    """
    Ana fonksiyon
    """
    print("RAG Pipeline - Sun Tzu Chatbot")
    
    # Test sorguları
    test_queries = [
        "savas strateji nasil belirlenir?",
        "dusman nasil yenilir?",
        "ordunun moralini nasil yukseltirsin?",
        "kaynaklari nasil yonetirsin?"
    ]
    
    model, embedder, collection = setup_rag_pipeline()
    
    if not all([model, embedder, collection]):
        return
    
    print("\nTest sorgulari calistiriliyor...")
    
    for query in test_queries:
        print(f"\n{'='*50}")
        response = get_response(model, embedder, collection, query)
        print(f"\nCevap: {response}")
    
    print(f"\n{'='*50}")
    print("Test tamamlandi!")

if __name__ == "__main__":
    main()
