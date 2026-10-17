"""Membuat dataset deteksi YOLO dari hasil ekspor COCO Roboflow yang disimpan LOKAL di raw_data.

Struktur raw_data (hasil ekstrak ZIP Roboflow):
    raw_data/train/_annotations.coco.json + gambar
    raw_data/valid/_annotations.coco.json + gambar
    raw_data/test/_annotations.coco.json  + gambar

Kelas di JSON Roboflow bernama "korban" dan "dummy".
Di dataset YOLO, "korban" diberi nama "victim" (sesuai tugas):
    0 = victim  (di JSON: korban)
    1 = dummy   (di JSON: dummy)

Langkah yang dilakukan (praktik persiapan data minggu 4):
  1. Mencari semua _annotations.coco.json di raw_data, lalu menggabungkan train+valid+test
     menjadi satu kumpulan (pembagian ulang nanti dilakukan skrip 05).
  2. [BARU] Membuang kotak DUPLIKAT: dua kotak sekelas pada gambar yang sama dengan
     tumpang tindih (IoU) di atas --iou dianggap duplikat; kotak pertama dipertahankan.
  3. Mengubah anotasi COCO (bbox piksel [x, y, w, h]) menjadi label YOLO
     (cx cy w h dinormalisasi) dengan nomor kelas baru 0, 1.
  4. Menyimpan hasil dalam struktur dataset_yolo/ (classes.txt, images/, labels/).
     Folder dataset_yolo/ yang lama SELALU dihapus lebih dulu, jadi menjalankan ulang
     skrip ini menimpa hasil lama (tidak menumpuk).

Pemakaian : python 00_ambil_subset_coco.py
            python 00_ambil_subset_coco.py --iou 1.0      (matikan pembuangan duplikat)
Lanjutan  : python 01_cek_anotasi.py dataset_yolo
"""
import argparse
import json
import random
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--kelas", nargs="+", default=["victim", "dummy"],
               help="nama kelas di dataset YOLO, urutan = nomor kelas (default: victim dummy)")
p.add_argument("--nama_json", nargs="+", default=["korban", "dummy"],
               help="nama kategori yang sesuai di file JSON, urutan sama dengan --kelas (default: korban dummy)")
p.add_argument("--jumlah", type=int, default=222, help="jumlah gambar maksimum (default 222)")
p.add_argument("--iou", type=float, default=0.5,
               help="batas IoU kotak duplikat (default 0.5; isi 1.0 untuk mematikan)")
p.add_argument("--raw", default="raw_data", help="folder data mentah (default raw_data)")
p.add_argument("--out", default="dataset_yolo", help="folder keluaran (default dataset_yolo)")
args = p.parse_args()
if len(args.kelas) != len(args.nama_json):
    raise SystemExit("Jumlah --kelas dan --nama_json harus sama.")
print(f"Kelas YOLO: {args.kelas} <- kategori JSON: {args.nama_json} | jumlah gambar maks: {args.jumlah}\n")


def iou(a, b):
    """IoU dua kotak COCO [x, y, w, h]."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    irisan = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - irisan
    return irisan / union if union > 0 else 0.0


RAW = Path(args.raw)
daftar_json = sorted(RAW.rglob("_annotations.coco.json"))
if not daftar_json:
    raise SystemExit(f"Tidak ada _annotations.coco.json di {RAW}/ (cek folder train/valid/test).")
print("File anotasi ditemukan:")
for j in daftar_json:
    print("  -", j.as_posix())

data = []          # (folder_gambar, coco)
for j in daftar_json:
    data.append((j.parent, json.loads(j.read_text(encoding="utf-8"))))

# Kategori dengan supercategory "none" adalah kategori kosong bawaan Roboflow, diabaikan.
ada = set()
for _, coco in data:
    for c in coco["categories"]:
        if c.get("supercategory") != "none":
            ada.add(c["name"].strip().lower())
print("\nKategori di JSON: " + ", ".join(sorted(ada)))
salah = [n for n in args.nama_json if n.lower() not in ada]
if salah:
    raise SystemExit(f"\nKategori tidak ada di JSON: {salah}\n"
                     f"Sesuaikan dengan --nama_json <nama1> <nama2> (nama persis seperti daftar di atas).")
nomor = {n.lower(): i for i, n in enumerate(args.nama_json)}   # nama JSON (huruf kecil) -> class_id YOLO

rekaman = []       # (path_gambar, W, H, [(class_id, bbox)])
abaikan, tidak_ada = 0, 0
duplikat = []      # (nama_file, nama_kelas) untuk laporan
for folder, coco in data:
    id_ke_nomor = {}
    for c in coco["categories"]:
        n = c["name"].strip().lower()
        if c.get("supercategory") != "none" and n in nomor:
            id_ke_nomor[c["id"]] = nomor[n]
    per_gambar = {}
    for a in coco["annotations"]:
        if a["category_id"] not in id_ke_nomor:
            abaikan += 1
        elif not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0:
            per_gambar.setdefault(a["image_id"], []).append((id_ke_nomor[a["category_id"]], a["bbox"]))
    for g in coco["images"]:
        path = folder / g["file_name"]
        if not path.exists():
            tidak_ada += 1
            continue
        # [BARU] buang kotak duplikat: sekelas dan IoU > batas dengan kotak yang sudah dipertahankan
        dipertahankan = []
        for cid, bbox in per_gambar.get(g["id"], []):
            if any(c == cid and iou(bbox, b) > args.iou for c, b in dipertahankan):
                duplikat.append((g["file_name"], args.kelas[cid]))
            else:
                dipertahankan.append((cid, bbox))
        rekaman.append((path, g["width"], g["height"], dipertahankan))

print(f"\nTotal gambar ditemukan : {len(rekaman)}")
if tidak_ada:
    print(f"PERINGATAN: {tidak_ada} gambar ada di JSON tetapi file-nya tidak ditemukan.")
if abaikan:
    print(f"PERINGATAN: {abaikan} kotak dari kategori lain diabaikan.")
if duplikat:
    print(f"Kotak duplikat dibuang  : {len(duplikat)}")
    for nama_file, kelas in duplikat:
        print(f"  - {kelas:7s} di {nama_file}")
if not rekaman:
    raise SystemExit("Tidak ada gambar yang bisa diproses.")

random.seed(0)
terpilih = random.sample(rekaman, min(args.jumlah, len(rekaman)))

OUT = Path(args.out)
if OUT.exists():
    shutil.rmtree(OUT)                          # hasil lama dihapus: dijalankan ulang = hasil terbaru
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)
(OUT / "classes.txt").write_text("\n".join(args.kelas) + "\n", encoding="utf-8")

hitung = {k: 0 for k in args.kelas}
latar, dipakai = 0, set()
for path, W, H, anotasi in terpilih:
    nama = path.name
    if nama in dipakai:                         # jaga-jaga bila ada nama file kembar antar split
        nama = f"{path.parent.name}_{nama}"
    dipakai.add(nama)
    baris = []
    for cid, (x, y, w, h) in anotasi:           # COCO: piksel, pojok kiri atas
        baris.append(f"{cid} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        hitung[args.kelas[cid]] += 1
    if not baris:
        latar += 1
    shutil.copy(path, OUT / "images" / nama)
    (OUT / "labels" / f"{Path(nama).stem}.txt").write_text("\n".join(baris) + ("\n" if baris else ""))

print(f"\nKelas baru      : {', '.join(f'{i} = {k}' for i, k in enumerate(args.kelas))}")
print(f"Gambar disalin  : {len(terpilih)}")
print(f"Gambar latar    : {latar}")
print("Jumlah kotak    : " + ", ".join(f"{k} {n}" for k, n in hitung.items()))
print(f"Siap di {OUT}/ -> lanjutkan dengan: python 01_cek_anotasi.py {OUT}")