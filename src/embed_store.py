"""
RAG Chatbot - Embedding ve Vector Database Kurulumu
Sun Tzu metinlerini embedding'e dönüştürüp ChromaDB'ye kaydetme
"""

from sentence_transformers import SentenceTransformer
import chromadb
import os

def setup_embedding_and_vector_db():
    """
    Sun Tzu metinlerini embedding'e dönüştürüp ChromaDB'ye kaydeder
    """
    print("Embedding ve Vector Database kurulumu basliyor...")
    
    # 1. Veri dosyasını oku
    print("Veri dosyasi okunuyor...")
    with open("data/sun_tzu.txt", "r", encoding="utf-8") as f:
        texts = [t.strip() for t in f.readlines() if t.strip()]
    
    print(f"Toplam {len(texts)} metin satiri bulundu")
    
    # 2. Embedding modeli (küçük ve hızlı bir Türkçe uyumlu model)
    print("Embedding modeli yukleniyor...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    print(f"Model yuklendi: {model_name}")
    
    # 3. Chroma client (kalıcı veritabanı)
    print("ChromaDB baglantisi kuruluyor...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # 4. Koleksiyon oluştur (varsa sil ve yeniden oluştur)
    collection_name = "sun_tzu_collection"
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Eski koleksiyon silindi: {collection_name}")
    except:
        pass
    
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Koleksiyon olusturuldu: {collection_name}")
    
    # 5. Her metin parçasını embed edip ekle
    print("Metinler embedding'e donusturuluyor ve kaydediliyor...")
    
    for i, text in enumerate(texts):
        if len(text) > 10:  # Çok kısa metinleri atla
            emb = model.encode(text).tolist()
            collection.add(
                ids=[str(i)],
                embeddings=[emb],
                documents=[text]
            )
            
            if (i + 1) % 10 == 0:
                print(f"   {i + 1}/{len(texts)} metin islendi...")
    
    print(f"Toplam {len(texts)} metin ChromaDB'ye kaydedildi!")
    
    return model, collection

def test_retriever(model, collection):
    """
    Retriever sistemini test eder
    """
    print("\nRetriever sistemi test ediliyor...")
    
    # Test sorguları
    test_queries = [
        "savas strateji nasil belirlenir?",
        "dusman nasil yenilir?",
        "ordunun moralini nasil yukseltirsin?",
        "kaynaklari nasil yonetirsin?"
    ]
    
    for query in test_queries:
        print(f"\nSorgu: '{query}'")
        query_emb = model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=2  # En alakalı 2 sonuç
        )
        
        print("En alakali sonuclar:")
        for j, doc in enumerate(results["documents"][0]):
            print(f"   {j+1}. {doc[:100]}...")
    
    print("\nRetriever testi tamamlandi!")

def main():
    """
    Ana fonksiyon
    """
    try:
        # Embedding ve vector database kurulumu
        model, collection = setup_embedding_and_vector_db()
        
        # Test sorgusu
        print("\nBasit test sorgusu...")
        query = "savas strateji nasil belirlenir?"
        query_emb = model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )
        
        print(f"\nSorgu: '{query}'")
        print("En alakali 3 sonuc:\n")
        for i, doc in enumerate(results["documents"][0]):
            print(f"{i+1}. {doc[:150]}...\n")
        
        # Detaylı test
        test_retriever(model, collection)
        
        print("\nEmbedding ve Vector Database kurulumu basariyla tamamlandi!")
        print("ChromaDB veritabani 'chroma/' klasorunde olusturuldu")
        print("RAG pipeline'inin retriever kismi hazir!")
        
    except Exception as e:
        print(f"Hata olustu: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
