---
name: foto-olcum
description: Use when reverse-engineering a machine part from photos into a parametric FreeCAD model with the foto-olcum MCP tools — measuring from backlit silhouettes, building the feature tree, comparing model to photo, and deciding which dimensions to ask the user to measure with calipers. Triggers - "fotoğraftan modelle", "tersine mühendislik", "parçanın fotoğrafı", "kumpasla doğrula", foto-olcum.
---

# Fotoğraftan parametrik model (foto-olcum)

Kullanıcıyla Türkçe konuş. Sayıları araçlardan al: fotoğrafa bakman şekli anlamak ve debug
görsellerini denetlemek içindir, pikselden ölçü tahmin etmek için değil. Her ölçünün bir
belirsizliği vardır; ölçüyü belirsizliğiyle birlikte söyle.

## Akış

1. Parçanın **görevini** sor: neye takılıyor, hangi yüzeyler eşleşiyor (pim, vida, klips, yuva).
   Kritik ölçüleri bu belirler. Prizmatik mi (Faz 1: 2.5D plaka) değil mi, ona bak.
2. `new_project`, `add_photos` (çerçeve + kamera adı), `process_photos`. Uyarıları oku:
   az marker, düşük kontrast, parça pencere kenarına değiyor, kalibrasyon yok.
3. **İlk kumpas ölçüsü kalınlıktır**: `set_measurement(id="thickness", source="caliper")`.
   Perspektif düzeltmesi buna bağlı.
4. `extract_features_2d` → debug görselini mutlaka incele: her köşe (V*) gerçek bir köşe mi,
   "?" işaretli köşe (açıklanamayan) çentik/pah/eğri mi, delik sayısı doğru mu, toz/gölge var mı.
5. Modeli kur: `suggested_model`'den başla, sonra **tasarım niyetiyle sadeleştir**:
   simetrik delikleri ortak ölçüye bağla (`"plate_w - h1_x"`), eşit radyusları tek ölçüye indir,
   paralel kenarları ifadeyle bağla. Her varsayımını kullanıcıya açıkça söyle
   ("R9.8'i R10 aldım", "iki delik aynı çapta varsaydım").
6. `build_freecad` → `render_and_compare`. Sapma bölgesi bir feature'a eşleniyorsa o feature'ın
   ölçüsünü düzelt; hiçbir feature'a yakın değilse modellenmemiş bir özellik vardır
   (Palet'teki gözden kaçan şerit gibi): fark görseline bak.
7. Kumpas turu (aşağıda) → `set_measurement` (kilitlenir) → tekrar build/compare.
   (Faz 2: harfli teknik resim + ölçü tablosu `make_drawing` henüz yok; şimdilik ölçüleri
   debug görselindeki V*/H*/L* adlarıyla ve net tanımlarla listede iste.)
8. Durma: tüm fotolarda maks sapma < 0.3 mm, kritik ölçüler kilitli, kontrol ölçüleri 0.2 mm içinde.
   İsteğe bağlı: önce ucuz bir 2D şablon bastırıp oturmayı dene, sonra 3D parça.

## Kumpas ölçüsü isteme

- **5–10 kritik ölçü** (gerekirse fazlası, sert sınır yok) + **2–3 kontrol ölçüsü**
  (`role="control"`). Kontrol ölçüleri hiçbir kalibrasyona girmez; sadece doğrulamadır.
  (Faz 2: otomatik kontrol/tutarlılık raporu `check_consistency` henüz yok; farkı sen hesapla.)
  Kalibrasyona giren bir ölçünün tutması kanıt sayılmaz.
- Seç: eşleşen yüzeyleri belirleyen ölçüler, belirsizliği hedefin üstünde olanlar, farklı
  malzeme/kenar tiplerinden en az birer ölçü, bir uzun boy (ölçek kontrolü).
- **Tanımı kesin yaz**: "sol uçtan 2. kulağın **iç** yüzüne", kumpasın hangi çenesiyle.
  Palet'teki en büyük hata kumpas değil, "nereden nereye" yanlış anlaşılmasıydı.
- Yarıçap ve sanal köşe gibi kumpasla ölçülemeyenler için "?" yazılabileceğini söyle.

## Ölçüm tahmine uymadığında

Tek tek düzeltmeden önce örüntüye bak:
- Tüm farklar **aynı oranda** → ölçek (çerçeve pencere ölçüsü girildi mi, kalibrasyon).
- Dış ölçüler hep büyük, delikler hep küçük (veya tersi), farklar **sabit mm** → kenar kayması
  (malzemeye bağlı, şişede opak kapakla camda farklıydı). Malzeme başına tek bir düzeltme.
  (Faz 2: `fit_edge_offsets` henüz yok; şimdilik kullanıcıya söyle ve etkilenen ölçüleri
  kumpasla kilitle.)
- Tek bir ölçü tutmuyor, komşuları tutuyor → büyük olasılıkla yanlış yerden ölçüldü:
  kibarca yeniden ölçmesini iste, nereden ölçeceğini tarif et.
- Bitişik iki özellik arası boşluk < 0.5 mm ya da negatif çıkıyorsa ölçüyü sorgula.

## Kullanıcının sözlü bilgisi

"Ani geçiş yok", "iki kulak aynı", "bu yüzey düz" gibi bilgileri kısıta çevir ve kaynağını
söyle. Yanlış kısıt gerçek bir detayı siler; emin değilsen sor.

## Sınırlar (şimdilik)

Faz 1 tek görüşlü 2.5D: dik duvarlı plaka + boydan boya delik + dik köşe radyusları. Cep
derinliği, pah, eğik duvar, dönel parça fotoğraftan çıkmaz; bunları kullanıcıya söyle ve
gerekirse kumpasla ölç.
