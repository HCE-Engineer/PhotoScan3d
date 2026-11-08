# foto-olcum

Prizmatik makine parçalarının fotoğraflarından **parametrik FreeCAD modeli** çıkaran MCP server.
Claude şekli yorumlar ve feature ağacını kurar; bu araç ölçer (her ölçü belirsizliğiyle),
FreeCAD modelini üretir, modeli fotoğrafla karşılaştırır. Kritik ölçüler kumpasla kilitlenir.

Tasarım: [docs/SPEC.md](docs/SPEC.md). Claude'un yöntemi: [skill/foto-olcum/SKILL.md](skill/foto-olcum/SKILL.md).

## Durum

| Faz | İçerik | Durum |
|---|---|---|
| 0 | MCP iskeleti, proje/ölçü modeli, kumpas kilidi, FreeCAD build (tablo + tam kısıtlı sketch + STEP/STL) | bitti |
| 1 | ChArUco altlık (ekran veya A4), kalibrasyon, arkadan aydınlatmalı silüet, alt-piksel kenar, perspektif (z=0/z=t), doğru/daire/radyus fit, belirsizlik, compare + fark görseli | bitti, **sadece sentetik fotoğraflarla doğrulandı** |
| 2 | Harfli teknik resim + ölçü tablosu, tutarlılık kontrolleri, malzeme kenar kayması (B) kalibrasyonu, snap, 2D şablon | sırada |
| 3–4 | Çok görüşlü prizmatik (cep, kanal, pah), yakın çekim | sonra |

Sentetik testlerde (3000×2250 px, ~0.15 mm/px, kamera 300–410 mm, A4 ChArUco altlık)
kalibrasyonlu ölçümlerde maks. hata ≈ 0.1 mm; kalibrasyonsuzda 10 mm kalın parça + 12° eğik çekimde 0.5 mm'ye kadar
(belirsizlik bunu raporluyor). Gerçek fotoğrafta malzemeye bağlı kenar kayması ek hata getirecek;
bu Faz 2'de kumpas ölçüleriyle kalibre edilecek.

## Kurulum

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -q
```

FreeCAD yolu varsayılan `~/Applications/FreeCAD_1.1.4.AppImage`; farklıysa `FOTO_OLCUM_FREECAD`.
Projeler `~/foto-olcum-projeler` altında (`FOTO_OLCUM_PROJECTS` ile değişir).

## Klasör taşındıysa

`.venv` mutlak yollar içerir; proje başka bir klasöre taşındıysa ortamı yeniden kur:

```bash
rm -rf .venv && python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]" && .venv/bin/python -m pytest -q
```

## Claude'a bağlama

Claude Code:
```bash
claude mcp add foto-olcum -s user -- /TAM/YOL/foto-olcum/.venv/bin/foto-olcum-mcp
```

Claude Desktop (`~/.config/Claude/claude_desktop_config.json`):
```json
{"mcpServers": {"foto-olcum": {
  "command": "/TAM/YOL/foto-olcum/.venv/bin/foto-olcum-mcp",
  "env": {"FOTO_OLCUM_FREECAD": "/home/huseyin/Applications/FreeCAD_1.1.4.AppImage"}}}}
```

Skill (Claude'un yöntemi) için: `cp -r skill/foto-olcum ~/.claude/skills/`

## İlk kullanım (bir kez)

İki altlık seçeneği var; ikisi de ChArUco deseni ve ortada parça için boş alan içerir.

**A) Ekran (önerilen): dizüstü ekranı hem desen hem arka ışık**

1. `make_reference_screen(connector="eDP-1")` → `~/foto-olcum-projeler/_altliklar/ekran14/`.
   Çözünürlük ve fiziksel boyut EDID'den okunur (14": 1920×1080, 309×174 mm → 30×16 kare,
   kare 10.3 mm, boş alan 268×124 mm).
2. Ekranı yatır (ekran yukarı), parlaklık en yüksek, gece ışığı ve otomatik parlaklık kapalı.
3. **Ölçek:** `eog --fullscreen ekran_olcum.png` → kırmızı halkalı A–B ve A–C köşe arası
   (nominal 133.90 mm) kumpasla → `set_sheet_measurement(sheet="ekran14", ...)`.
4. **Kalibrasyon:** `eog --fullscreen ekran_cekim.png`, parça yokken 15–20 foto (farklı açılar,
   desen bazen kadraj köşelerinde) → `calibrate_camera(sheet="ekran14")`.

Pikseller camın ~0.8 mm altındadır (`glass_mm`); bu modellenir (sentetikte yok sayılınca hata
0.05 → 0.26 mm).

**B) A4 kağıt:** `make_reference_sheet` → `a4.pdf`, %100 / gerçek boyutta bas, A–B / A–C
(127.0 mm) kumpasla → `set_sheet_measurement`, kalibrasyon `calibrate_camera(sheet="a4")`.
Kağıt arkadan aydınlatılmış bir yüzeyde (monitör, ışıklı tablet) kullanılır.

## Çekim

`ekran_cekim.png` tam ekran açıkken parçayı ortadaki beyaz alana koy (oturduğu yüz aşağıda,
desene değmeden). Telefon ~30–40 cm yukarıda, yaklaşık dik, kalibrasyondaki zoom; desenin
mümkün olduğunca çoğu kadrajda. Fotoğrafı özgün çözünürlükte dosya olarak ver.

## Akış

`new_project` → `add_photos` → `process_photos` → kalınlığı kumpasla `set_measurement(id="thickness")`
→ `extract_features_2d` (debug görseline bak) → `set_model` (önerilen modelden başla) →
`build_freecad` → `render_and_compare` → kritik ölçüler kumpasla → `set_measurement` → tekrar.

## Bilinen sınırlar

- Doğrulama yalnızca sentetik fotoğraflarla yapıldı. Gerçek telefon fotoğrafında beyaz ekran
  parlaması, keskinleştirme haleleri ve JPEG ek kenar kayması getirir; `EDGE_BIAS_PX = 0.5`
  bir tahmindir. Gerçek hata ancak ölçüleri bilinen bir test plakasıyla görülecek.
- `extract_features_2d` her çalıştığında o fotonun parça koordinat sistemini (`part_frame`)
  yeniler. Build'den sonra başka eksen/orijinle yeniden çıkarırsan modeli de yeniden kur,
  yoksa `render_and_compare` kayık karşılaştırır.
