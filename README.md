# foto-olcum

Prizmatik makine parçalarının fotoğraflarından **parametrik FreeCAD modeli** çıkaran MCP server.
Claude şekli yorumlar ve feature ağacını kurar; bu araç ölçer (her ölçü belirsizliğiyle),
FreeCAD modelini üretir, modeli fotoğrafla karşılaştırır. Kritik ölçüler kumpasla kilitlenir.

Tasarım: [docs/SPEC.md](docs/SPEC.md). Claude'un yöntemi: [skill/foto-olcum/SKILL.md](skill/foto-olcum/SKILL.md).

## Durum

| Faz | İçerik | Durum |
|---|---|---|
| 0 | MCP iskeleti, proje/ölçü modeli, kumpas kilidi, FreeCAD build (tablo + tam kısıtlı sketch + STEP/STL) | bitti |
| 1 | Referans çerçeve, kalibrasyon, arkadan aydınlatmalı silüet, alt-piksel kenar, perspektif (z=0/z=t), doğru/daire/radyus fit, belirsizlik, compare + fark görseli | bitti, **sadece sentetik fotoğraflarla doğrulandı** |
| 2 | Harfli teknik resim + ölçü tablosu, tutarlılık kontrolleri, malzeme kenar kayması (B) kalibrasyonu, snap, 2D şablon | sırada |
| 3–4 | Çok görüşlü prizmatik (cep, kanal, pah), yakın çekim | sonra |

Sentetik testlerde (3000×2250 px, ~0.15 mm/px, kamera 300–410 mm) kalibrasyonlu ölçümlerde
maks. hata ≈ 0.1 mm; kalibrasyonsuzda 10 mm kalın parça + 12° eğik çekimde 0.5 mm'ye kadar
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

1. **Çerçeveyi bas:** `make_reference_frame` → `~/foto-olcum-projeler/_cerceveler/cerceve_a/`.
   Açık renk PLA, %100 dolgu; 0.6 mm'de siyaha filament değişimi en iyisi (tek renk de çalışır).
   Varsayılan dış ölçü 210×210 mm, pencere 138×138 mm; tablana göre değiştir.
2. **Pencereyi ölç:** basılan çerçevenin pencere iç genişlik/yüksekliğini kumpasla →
   `set_frame_measurement`.
3. **Telefonu kalibre et:** `make_calibration_pattern` → PNG'yi monitörde tam ekran aç,
   ölçümde kullanacağın zoom ile 15–20 fotoğraf çek → `calibrate_camera`.

## Çekim

Tablet/monitör beyaz ekran, üstünde çerçeve, parça pencerede (oturduğu yüz aşağıda, kenarlara
değmeden). Telefon ~40 cm yukarıda, yaklaşık dik, kalibrasyondaki zoom. Fotoğrafı özgün
çözünürlükte dosya olarak ver.

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
