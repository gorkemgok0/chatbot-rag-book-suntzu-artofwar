"""
RAG Chatbot - Veri Seti Hazırlama
Sun Tzu'nun Savaş Sanatı veri setini Hugging Face'ten indirip işleme
"""

from datasets import load_dataset
import os
import json

def download_and_prepare_sun_tzu_data():
    """
    Sun Tzu veri setini indirir ve işler
    """
    print("Sun Tzu veri seti indiriliyor...")
    
    try:
        # 1. Dataset'i Hugging Face'ten yükle - farklı formatları dene
        print("Dataset yukleniyor...")
        
        # Önce veri setinin mevcut olup olmadığını kontrol et
        try:
            dataset = load_dataset("umithy/sun_tzu_savas_sanati", split="train")
            print("Dataset basariyla yuklendi!")
        except Exception as e:
            print(f"Dataset yuklenirken hata: {e}")
            print("Alternatif yontem deneniyor...")
            
            # Alternatif olarak, veri setini JSON formatında yuklemeyi dene
            dataset = load_dataset("umithy/sun_tzu_savas_sanati", split="train", streaming=True)
            print("Streaming modunda dataset yuklendi!")
        
        # 2. Dataset içeriğini kontrol et
        print("\nDataset bilgileri:")
        print(f"Dataset tipi: {type(dataset)}")
        
        # 3. İlk birkaç örneği incele
        print("\nIlk ornek:")
        first_item = next(iter(dataset))
        print(first_item)
        
        return dataset, first_item
        
    except Exception as e:
        print(f"Hata: {e}")
        print("Manuel olarak Sun Tzu metni olusturuluyor...")
        return create_manual_sun_tzu_data()

def create_manual_sun_tzu_data():
    """
    Manuel olarak Sun Tzu metni oluşturur (Hugging Face dataset çalışmazsa)
    """
    print("Manuel Sun Tzu metni olusturuluyor...")
    
    # Sun Tzu'nun Savaş Sanatı'nın temel bölümleri
    sun_tzu_content = """
=== SUN TZU'NUN SAVAŞ SANATI ===

BÖLÜM 1: SAVAŞ PLANLAMA

Savaş bir sanattır. Savaşmak zorunda kalırsan, kazanmanın tek yolu savaşı önceden planlamaktır. Savaştan önce, düşmanını tanı, kendi gücünü değerlendir ve koşulları analiz et.

Savaşta en iyi strateji, savaşmadan kazanmaktır. Eğer savaşmak zorundaysan, hızlı ve etkili bir şekilde zafer kazan.

BÖLÜM 2: SAVAŞ YÖNETİMİ

Savaşın maliyeti yüksektir. Uzun süren savaşlar hem ekonomik hem de insani açıdan yıkıcıdır. Bu yüzden savaşları mümkün olduğunca kısa tutmak gerekir.

Kaynaklarını akıllıca kullan. Düşmanın kaynaklarını tüket, kendi kaynaklarını koru.

BÖLÜM 3: STRATEJİK SALDIRI

En iyi savunma saldırıdır. Ancak saldırıdan önce düşmanın zayıf noktalarını tespit et.

Düşmanın moralini boz, onları şaşırt ve beklenmedik yerden saldır.

BÖLÜM 4: TAKTİKSEL DÜZENLEME

Ordunu iyi düzenle. Disiplin, eğitim ve organizasyon zaferin anahtarıdır.

Komutan olarak, askerlerinin güvenini kazan. Onları hem ödüllendir hem de cezalandır.

BÖLÜM 5: ENERJİ YÖNETİMİ

Savaşta enerjini akıllıca kullan. Gereksiz çatışmalardan kaçın, gücünü doğru zamanda doğru yerde kullan.

Momentumunu koru. Bir kez üstünlük sağladığında, düşmanı takip et ve zafere ulaş.

BÖLÜM 6: ZAYIFLIK VE GÜÇLÜLÜK

Düşmanının güçlü yanlarını atla, zayıf yanlarına saldır. Kendi güçlü yanlarını koru, zayıf yanlarını güçlendir.

Esneklik göster. Koşullara göre stratejini değiştir.

BÖLÜM 7: MANEVRALAR

Savaşta manevra yapabilmek çok önemlidir. Düşmanı yanılt, onları beklenmedik yönlere çek.

Hızlı hareket et, ama dikkatli ol. Acele etme, ama gecikme de.

BÖLÜM 8: TAKTİK DEĞİŞİKLİKLERİ

Savaş dinamiktir. Koşullar değiştiğinde, taktiğini de değiştir.

Düşmanın stratejisini öğren ve buna göre karşı önlem al.

BÖLÜM 9: ORDU HAREKETLERİ

Ordunu hareket ettirirken dikkatli ol. Düşmanın seni görmesini engelle.

Farklı yollardan git, beklenmedik yerlerden saldır.

BÖLÜM 10: ARAZİ

Araziyi iyi tanı. Dağlar, nehirler, ormanlar - hepsi senin avantajın olabilir.

Araziyi kullanarak düşmanı tuzağa düşür.

BÖLÜM 11: DOKUZ ARAZİ TÜRÜ

Farklı arazi türleri farklı taktikler gerektirir. Her arazi türünde nasıl savaşacağını bil.

Kendi toprağında savaşırken farklı, düşman toprağında savaşırken farklı strateji uygula.

BÖLÜM 12: ATEŞ SALDIRILARI

Ateşi silah olarak kullan. Düşmanın erzaklarını, silahlarını ve moralini yak.

Ancak ateşi kullanırken dikkatli ol - kendi askerlerini de yakma.

BÖLÜM 13: CASUSLUK VE BİLGİ

Bilgi savaşta en önemli silahtır. Düşman hakkında mümkün olduğunca çok bilgi topla.

Casusları akıllıca kullan. Hem kendi casuslarını koru, hem de düşmanın casuslarını tespit et.

=== SUN TZU'NUN ÖĞÜTLERİ ===

• Savaştan önce düşmanını tanı
• Kaynaklarını akıllıca kullan  
• Hızlı hareket et ama dikkatli ol
• Esneklik göster
• Bilgi topla ve kullan
• Moralini koru, düşmanın moralini boz
• Zaferi en az kayıpla kazanmaya çalış
"""
    
    # data klasörünü oluştur
    os.makedirs("data", exist_ok=True)
    
    # Veriyi dosyaya kaydet
    with open("data/sun_tzu.txt", "w", encoding="utf-8") as f:
        f.write(sun_tzu_content)
    
    print("Manuel Sun Tzu metni basariyla data/sun_tzu.txt dosyasina kaydedildi!")
    return None, "Manuel metin olusturuldu"

def process_dataset_to_file(dataset, first_item):
    """
    Dataset'i dosyaya işler
    """
    # data klasörünü oluştur (yoksa)
    os.makedirs("data", exist_ok=True)
    
    # 4. Veriyi .txt dosyasına kaydet
    print("\nVeri dosyaya kaydediliyor...")
    with open("data/sun_tzu.txt", "w", encoding="utf-8") as f:
        try:
            if hasattr(dataset, '__iter__'):
                for i, item in enumerate(dataset):
                    if isinstance(item, dict):
                        # Farklı anahtar isimlerini dene
                        content = item.get('content', item.get('text', item.get('metin', str(item))))
                        chapter = item.get('chapter', item.get('bolum', f'Bolum {i+1}'))
                        f.write(f"=== BOLUM {i+1}: {chapter} ===\n\n")
                        f.write(content.strip() + "\n\n")
                    else:
                        f.write(f"=== BOLUM {i+1} ===\n\n")
                        f.write(str(item).strip() + "\n\n")
        except Exception as e:
            print(f"Dosyaya yazarken hata: {e}")
            return False
    
    print("Veri basariyla data/sun_tzu.txt dosyasina kaydedildi!")
    
    # 5. Küçük bir kontrol
    print("\nDosya kontrolu (ilk 500 karakter):")
    with open("data/sun_tzu.txt", "r", encoding="utf-8") as f:
        text = f.read(500)
        print(text)
        print("...")
    
    # Dosya boyutu bilgisi
    file_size = os.path.getsize("data/sun_tzu.txt")
    print(f"\nDosya boyutu: {file_size:,} byte ({file_size/1024:.1f} KB)")
    
    return True

if __name__ == "__main__":
    dataset, first_item = download_and_prepare_sun_tzu_data()
    
    # Eğer dataset başarıyla yüklendiyse, dosyaya işle
    if dataset is not None:
        success = process_dataset_to_file(dataset, first_item)
        if not success:
            print("Dataset isleme basarisiz oldu, manuel metin kullaniliyor.")
    else:
        print("Manuel metin kullanildi.")
