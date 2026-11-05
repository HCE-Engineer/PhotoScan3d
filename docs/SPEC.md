# foto-olcum — Fotoğraftan Parametrik FreeCAD Modeli (MCP Server + Skill)

Sürüm 2 (06.11.2026). İlk spec'in Hüseyin'le konuşulup iki gerçek deneyim raporuyla
(şişe profili: fotoğraftan çıkarma, Palet: STL → parametrik model + kumpas) güncellenmiş hali.
`[KARAR]` işaretli maddeler yazılım tarafında Claude'un verdiği kararlardır, itiraz edilebilir.

## 1. Amaç ve neden

Hüseyin tersine mühendislik yapıyor: bugün parçanın üstten/yandan dik fotoğraflarını çekip
ölçekliyor ve üstüne elle çiziyor. Yavaş ve yorucu. Bu araç o işi Claude'a (Opus, Claude
aboneliğiyle) yaptırır:

1. Hüseyin fotoğrafları atar.
2. Claude şekli yorumlar, feature ağacını kurar; bu araç ölçer, FreeCAD modelini üretir,
   modeli fotoğraflarla karşılaştırır.
3. Claude harfli bir teknik resim + ölçü tablosu verir; Hüseyin kritik ölçüleri kumpasla doldurur.
4. Model güncellenir; (isteğe bağlı) önce ucuz bir 2D şablon, sonra parça basılır.

**Başarı:** basılan parça/karşı parça oturuyor. Sayısal hedef: kritik ölçüler kumpasla kilitli,
geri kalanlar ±0.3 mm içinde ve her ölçünün belirsizliği dürüstçe raporlanmış.

Bu bir uygulama değil, **ajan için araç**: MCP server (ölçme, model üretme, karşılaştırma) +
bir skill (Claude'un karar verme yöntemi: hangi ölçü istenir, hata ölçek mi kayma mı yanlış
ölçüm mü). Claude piksel saymaz, sayıları araçtan alır; fotoğrafa bakması şekli anlamak ve
debug görsellerini kontrol etmek içindir.

## 2. Kapsam

- **Şimdi (Faz 1–3):** prizmatik makine parçaları: plaka, braket, blok, delik, cep, kanal, pah, radyus.
- **Sonra (kapsam dışı değil, ertelendi):** dönel parçalar (`revolve`, şişe gibi), bombeli
  yüzeyler (`loft`). Veri modeli ve compare bunlara kapalı tasarlanmaz.
- **Kapsam dışı:** vida dişi geometrisi (sadece "M6" etiketi), serbest form tarama/nokta bulutu.
- Platform: **Ubuntu**, lokal. Claude Desktop / Claude Code stdio MCP.

## 3. Ortam `[KARAR]`

| Konu | Karar |
|---|---|
| Python | Proje içi `.venv` (python3.12 -m venv). `uv` kurulu değil, şart değil; `pyproject.toml` uv uyumlu. `~/muhendislik` ortamına dokunulmaz (OpenCV yok, FreeCAD zaten ayrı süreç). |
| OpenCV | `opencv-contrib-python-headless` 5.x (`cv2.aruco.CharucoDetector`, `cv2.calibrateCamera`) |
| MCP | `mcp` 2.x, `MCPServer` (FastMCP'nin yeni adı) |
| FreeCAD | `~/Applications/FreeCAD_1.1.4.AppImage freecadcmd <betik.py>`; yol `FOTO_OLCUM_FREECAD` env'inden. İş dosyası yolu env ile verilir (ek argümanlar dosya sanılabilir). `freecadcmd -c` kullanılmaz (Body oluştururken çöküyor). Betik `result.json` yazar; yoksa başarısız sayılır. |
| stdout | MCP stdio olduğu için sunucu stdout'a hiçbir şey yazmaz; log stderr'e. FreeCAD çıktısı yakalanır. |

## 4. Referans ve kalibrasyon `[KARAR]`

Hüseyin'in kağıt yazıcısı yok, 3D yazıcısı ve monitörü var. Bu yüzden:

- **Kamera kalibrasyonu:** ChArUco deseni PNG olarak üretilir, **monitörde tam ekran** gösterilir,
  telefonla 15–20 açıdan çekilir. Kamera iç parametreleri (K, distorsiyon) desenin fiziksel
  boyutundan bağımsızdır, bu yüzden monitör yeterli. Kalibrasyon yoksa: markerlardan tek
  görüntüde odak uzaklığı tahmini + distorsiyon sıfır, uyarıyla.
- **Referans çerçeve (3D baskı):** ortası boş, kenarlarında ArUco marker'lar olan plaka.
  Arkadan aydınlatmada (monitör/tablet beyaz ekran) parça ortadaki pencereye konur.
  - Marker'ın beyaz hücreleri 0.4 mm ince (ışık geçer), siyah hücreleri kalın. Tek renk açık
    renkli PLA ile arkadan ışıkta kontrast verir; 0.4 mm'de siyaha filament değişimi en iyisi.
  - Baskı ölçüsü ±0.1–0.2 mm ve yazıcının XY ölçeği %0.2–0.5 sapabilir: çerçevenin
    **pencere iç genişliği ve yüksekliği kumpasla ölçülüp** girilir, marker koordinatları buna
    göre ölçeklenir. Nominal CAD sayısından ±0.1 mm çıkmaz.
  - Gerçek baskının fotoğrafında marker algılama test edilmeden üstüne bina edilmez.
- Bozuk para / rulman: yedek ölçek doğrulaması (bilinen çap).

## 5. Çekim protokolü (araç anlatabilmeli)

1. Kalibrasyon (telefon başına bir kez, aynı zoom ve odak modu).
2. **Silüet (2.5D ana yöntem):** beyaz ekran + çerçeve, parça pencerede, oturduğu yüz aşağıda.
   Kamera yaklaşık dik, ≥40 cm uzaktan + zoom. Parçanın tamamı ışıklı fonun önünde.
3. Genel 3D çekim (Faz 3), yakın çekim (Faz 4): her karede marker.
4. Özgün çözünürlükte dosya (sohbete yapıştırma küçültür). Zoom kalibrasyondan farklıysa uyarı.

## 6. Ölçüm modeli (raporlardan öğrenilenler)

- **Kenar kayması (B) malzemeye bağlıdır** ve en büyük hata kaynağıdır (şişede opak kapakla
  cam arasında ~2 px, 0.4 mm). Kenar: dış yarı-maksimum gradyan / %50 geçiş, alt-piksel.
  Kumpas ölçüleri gelince malzeme başına B en küçük karelerle bulunur:
  dış ölçü `= foto + 2B`, delik çapı `= foto − 2B`.
- **Kumpas ölçülerinin 2–3'ü kalibrasyona girmez, sadece doğrulama (kontrol ölçüsü).**
  Kalibrasyona giren ölçünün tutması kanıt değildir.
- **Silüet perspektifi:** dik duvarlı 2.5D parçada kontur noktası hem z=0 hem z=t düzlemine
  izdüşürülür; gerçek iz, malzeme dış normaline göre **içte kalan** olandır. t belirsizliği
  belirsizliğe eklenir. Kalınlık bilinmiyorsa ilk istenen kumpas ölçüsü t'dir.
- **Belirsizlik zorunlu:** fit standart hatası ⊕ kenar kayması payı ⊕ ölçek (poz) ⊕
  perspektif/kalınlık. Belirsizliksiz ölçü döndüren fonksiyon yok.
- Ham kenar noktaları + dönüşüm parametreleri saklanır; yeni ölçü gelince her şey baştan hesaplanır.
- Kullanıcının sözlü bilgisi ("ani geçiş yok", "iki kulak aynı") kısıta çevrilebilir; yanlış
  kısıt gerçek detayı siler, bu yüzden kaynağı kaydedilir.

## 7. Kumpas döngüsü

- **5–10 kritik ölçü** (parçanın görevine göre: neye takılıyor, hangi yüzeyler eşleşiyor;
  gerekirse 10'u geçebilir, sert sınır yok) **+ 2–3 kontrol ölçüsü.**
- Teknik resim (HLR görünüşler, harfli ölçüler) ve ölçü tablosu **aynı koddan** üretilir.
  Her satır: harf, görünüş, tanım ("sol uçtan 2. kulak **iç** yüzüne"), mevcut değer,
  ölçülen (boş), tür (kritik/kontrol), model parametresi eşlemesi.
- Doldurulmuş tablo **asla üzerine yazılmaz** (`olcu_tablosu_dolu_<tarih>.csv`).
- **Tutarlılık kontrolleri:** kontrol ölçüsü farkı > 0.2 mm, bitişik özellikler arası boşluk
  < 0.5 mm veya negatif, duvar < 0.8 mm (FDM) → uyarı ve "şunu yeniden ölç" isteği.
- Ölçülemeyenler (yarıçap, sanal köşe) için "?" girilebilir.
- Kilitli (kumpas) ölçü hiçbir iterasyonda değişmez; snap kumpası ezmez.
- Baskı sonrası geri bildirim ("X'te sıkı") parametre güncellemesine dönüşür.
- İsteğe bağlı: önce 2D negatif şablon STL, sonra parça.

## 8. Mimari

```
src/foto_olcum/
  server.py        MCPServer tool tanımları (stdout'a yazmaz)
  config.py        yollar, FreeCAD yolu
  project.py       proje klasörü, measurements.json, model.json, log.jsonl
  frame.py         referans çerçeve: marker yerleşimi, STL üretimi, ölçülen pencereyle ölçekleme
  calib.py         ChArUco deseni, kalibrasyon, kalibrasyonsuz odak tahmini, poz (solvePnP)
  geometry.py      görüntü ↔ düzlem (z=t), perspektif iz seçimi
  segment.py       arkadan aydınlatmalı silüet maskesi
  edges.py         alt-piksel kenar
  fit.py           doğru/daire (TLS, Taubin) + kontur bölütleme + belirsizlik
  features2d.py    foto başına 2D özellik çıkarma (dış kontur, delikler), mm + belirsizlik
  model.py         feature ağacı doğrulama
  build.py         FreeCAD alt süreç çağrısı
  freecad_build/build_fc.py   freecadcmd içinde çalışan betik
  compare.py       mesh → kamera → silüet; IoU, sapma, fark görseli
  synth.py         sentetik test görüntüsü (compare'den bağımsız projeksiyon kodu)
skill/SKILL.md     Claude'un yöntemi (Faz 2)
```

## 9. Veri modeli

Proje klasörü (`FOTO_OLCUM_PROJECTS`, varsayılan `~/foto-olcum-projeler`):
```
<parca>/ photos/raw  debug/  masks/  measurements.json  model.json  out/  log.jsonl
```

`measurements.json` → `{id: {label, value_mm, uncertainty_mm, source, locked, evidence, feature_ref, material}}`
`source ∈ photo_coarse | photo_closeup | caliper | inferred | snapped`.

`model.json`:
```json
{"units": "mm", "features": [
  {"id": "plate", "type": "pad", "plane": "XY",
   "profile": [["0","0"], ["plate_w","0"], ["plate_w","plate_h"], ["0","plate_h"]],
   "length": "thickness"},
  {"id": "hole_1", "type": "hole", "plane": "XY", "center": ["hole_1_x","hole_1_y"],
   "diameter": "hole_1_dia", "depth": "through", "thread": "M6"},
  {"id": "r1", "type": "fillet", "edges": [["plate", 2]], "radius": "corner_r"}
]}
```
Sayılar ölçü id'si, ifade (`"plate_w - 5"`) ya da sabit olabilir. Desteklenen tipler:
`pad, pocket, hole, slot, fillet, chamfer, pattern_linear, pattern_polar`
(sonra: `revolve, loft`). Faz 1'de: `pad, pocket, hole, fillet`.

FreeCAD çıktısı: `Params` Spreadsheet (alias = ölçü id; ek sütunlar ±, kaynak, kilit),
tüm kısıtlar expression ile bağlı, sketch'ler tam kısıtlı, STEP + STL.

## 10. MCP tool'ları

| Tool | Faz |
|---|---|
| `new_project`, `status` | 0 |
| `make_calibration_pattern`, `calibrate_camera` | 1 |
| `make_reference_frame`, `set_frame_measurement` | 1 |
| `add_photos`, `process_photos`, `extract_features_2d`, `measure` | 1 |
| `set_measurement` (değer, kilit, malzeme) | 1 |
| `set_model`, `build_freecad`, `render_and_compare` | 1 |
| `make_drawing` (harfli teknik resim + tablo), `import_measurement_table`, `check_consistency`, `fit_edge_offsets`, `suggest_snaps`, `make_gauge` | 2 |
| `request_closeup`, çok görüşlü poz | 3–4 |

**Durma kriteri:** tüm silüetlerde maks sapma < 0.3 mm **ve** kritik ölçülerin hepsi kilitli
**ve** kontrol ölçüleri 0.2 mm içinde.

## 11. Fazlar

- **Faz 0:** iskelet, MCP server, `new_project`/`status`, FreeCAD smoke build (STEP döner).
- **Faz 1:** çerçeve STL, kalibrasyon, backlit silüet, z=t düzlemi + perspektif iz seçimi,
  doğru/daire fit, belirsizlik, plaka + delik → FreeCAD + Spreadsheet + STEP, compare + fark görseli.
- **Faz 2:** teknik resim + tablo, tutarlılık kontrolleri, malzeme B kalibrasyonu, snap, 2D şablon, SKILL.md.
- **Faz 3:** çok görüşlü prizmatik (cep, kanal, pah, radyus).
- **Faz 4:** yakın çekim inceltme. **Sonra:** dönel/bombeli.

## 12. Test

- Sentetik önce: bilinen parça → **compare'den bağımsız** bir projeksiyon koduyla, süper
  örnekleme + küçültme (kenar yumuşaması gerçekçi olsun), gürültü/blur → pipeline → gerçek
  değerlerle karşılaştır, belirsizlik gerçek hatayı kapsıyor mu?
- Birim: Taubin, homografi, perspektif iz seçimi, model doğrulama.
- Gerçek: ölçüleri bilinen delikli test plakası (3D baskı). Kabul: kumpassız ölçülerin %90'ı ±0.3 mm.

## 13. Bağlantı (örnek, proje kalıcı bir klasöre taşındıktan sonra)

```json
{"mcpServers": {"foto-olcum": {
  "command": "/yol/foto-olcum/.venv/bin/foto-olcum-mcp",
  "env": {"FOTO_OLCUM_FREECAD": "/home/huseyin/Applications/FreeCAD_1.1.4.AppImage"}}}}
```
