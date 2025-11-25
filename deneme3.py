import csv
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import mysql.connector
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog

# ====================== MySQL BAĞLANTISI ======================
DB_CFG = dict(
    host="localhost",
    user="root",
    password="gmz7042",  
    database="gamze_minikod"
)

try:
    db = mysql.connector.connect(**DB_CFG)
    cursor = db.cursor()
    print("Veritabani baglantisi basarili")
except mysql.connector.Error as e:
    print(f"Veritabani baglanti hatasi: {e}")
    db = None
    cursor = None

PROJECT_DIR = Path(__file__).resolve().parent
FOTO_KLASOR = PROJECT_DIR / "ogrenciler"
MODEL_DOSYA = PROJECT_DIR / "face_model.pkl"
YOKLAMA_CSV = PROJECT_DIR / "yoklama_log.csv"
ATTENDANCE_WINDOW = "Yuz Tanima - Yoklama"
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["Isim", "Tarih", "Saat"]
FOTO_KLASOR.mkdir(parents=True, exist_ok=True)


def load_known_faces(image_dir: Path):
    encodings = []
    names = []
    if not image_dir.exists():
        return encodings, names
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_EXT:
            continue
        try:
            image = face_recognition.load_image_file(str(image_path))
            locs = face_recognition.face_locations(image)
            if not locs:
                print(f"Uyarı: '{image_path.name}' dosyasında yüz bulunamadı, atlandı.")
                continue
            encoding = face_recognition.face_encodings(image, locs)[0]
            encodings.append(encoding)
            raw_name = image_path.stem
            names.append(raw_name)
            display_name = raw_name.replace("_", " ").strip() or raw_name
            print(f"'{image_path.name}' için yüz kaydedildi ({display_name}).")
        except Exception as exc:
            print(f"'{image_path.name}' yüklenemedi: {exc}")
    return encodings, names

ENG2TR = {
    "Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
    "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"
}

# Global değişkenler
known_face_encodings = []
known_face_names = []

# Kamera & Recognition (Tk tabanlı fotoğraf kaydı)
camera_active = False
camera_thread = None
current_frame = None
camera_label = None
camera_window = None
captured_frame = None

recognition_active = False
recognition_thread = None
recognition_frame = None
recognition_lock = threading.Lock()
last_recognized_name = "Bilinmiyor"
last_recognized_time = 0.0

# OpenCV tabanlı yoklama oturumu
attendance_session_active = False
attendance_thread = None

# Tk root referansı
root = None

# ====================== MODEL İŞLEMLERİ ======================
def ensure_csv_header(path: Path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def model_egit_ve_kaydet():
    global known_face_encodings, known_face_names
    encodings, names = load_known_faces(FOTO_KLASOR)

    if not encodings:
        raise RuntimeError("En az bir yüz fotoğrafı ekleyin.")
    
    known_face_encodings = encodings
    known_face_names = names
    
    with MODEL_DOSYA.open("wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"Model kaydedildi: {len(names)} öğrenci")

def model_yukle():
    global known_face_encodings, known_face_names
    if MODEL_DOSYA.exists():
        try:
            with MODEL_DOSYA.open("rb") as f:
                data = pickle.load(f)
            known_face_encodings = data.get("encodings", [])
            known_face_names = data.get("names", [])
            print(f"Model yüklendi: {len(known_face_names)} öğrenci")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            known_face_encodings, known_face_names = [], []
    else:
        print("Model yok, eğitiliyor...")
        try:
            model_egit_ve_kaydet()
        except Exception as e:
            print(f"Eğitim hatası: {e}")
    if not known_face_encodings:
        print("Pickle modeli boş geldi, klasörden yükleniyor...")
        known_face_encodings, known_face_names = load_known_faces(FOTO_KLASOR)
        print(f"Klasörden yüklenen öğrenci sayısı: {len(known_face_names)}")

# ====================== VERİTABANI FONKSİYONLARI ======================
def bugunun_tr_gunu():
    return ENG2TR.get(datetime.now().strftime("%A"), "Bilinmiyor")

def ogrenci_id_bul(ad, soyad):
    if not cursor: return None
    try:
        cursor.execute("SELECT id FROM ogrenciler WHERE ad=%s AND soyad=%s", (ad, soyad))
        r = cursor.fetchone()
        return r[0] if r else None
    except Exception as e:
        print(f"ID bulma hatası: {e}")
        return None

def yoklama_kaydi_ekle_ve_sayac(ogrenci_id, durum, tarih, saat):
    if not cursor or not db: return False
    try:
        cursor.execute("SELECT id, durum FROM yoklama WHERE ogrenci_id=%s AND tarih=%s", (ogrenci_id, tarih))
        existing = cursor.fetchone()
        if not existing:
            cursor.execute("INSERT INTO yoklama (ogrenci_id, tarih, durum, saat) VALUES (%s,%s,%s,%s)",
                          (ogrenci_id, tarih, durum, saat))
            db.commit()
            return True
        else:
            record_id, old_durum = existing
            if old_durum != durum:
                cursor.execute("UPDATE yoklama SET durum=%s, saat=%s WHERE id=%s", (durum, saat, record_id))
                db.commit()
                return True
        return False
    except Exception as e:
        print(f"Yoklama hatası: {e}")
        return False

# ====================== YARDIMCI - YOKLAMA KAYDI ======================
def log_attendance_hit(name):
    ensure_csv_header(YOKLAMA_CSV)
    try:
        parts = name.split("_", 1)
        ad = parts[0]
        soyad = parts[1] if len(parts) > 1 else ""
        ogr_id = ogrenci_id_bul(ad, soyad)
        if not ogr_id:
            print(f"{name} için öğrenci bulunamadı.")
            return
        dt = datetime.now()
        tarih = dt.strftime("%Y-%m-%d")
        saat = dt.strftime("%H:%M:%S")
        yoklama_kaydi_ekle_ve_sayac(ogr_id, "geldi", tarih, saat)
        with YOKLAMA_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, tarih, saat])
        print(f"{name} yoklaması alındı")
    except Exception as e:
        print(f"Yoklama kaydı hatası: {e}")


# ====================== KAMERA & TANIMA (Tk için) ======================
def camera_thread_function(cap):
    global current_frame, camera_active, recognition_frame
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        # frame BGR, current_frame için RGB versiyonunu sakla
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Recognizer’a tam frame veriyoruz (BGR)
        with recognition_lock:
            recognition_frame = frame.copy()

        # Ekran için RGB tutuyoruz
        current_frame = frame_rgb
    cap.release()


def recognition_worker(mode="attendance"):
    global recognition_active, recognition_frame, current_frame, last_recognized_name, last_recognized_time

    ensure_csv_header(YOKLAMA_CSV)

    process_this_frame = True

    while recognition_active:
        frame_for_recog = None
        with recognition_lock:
            if recognition_frame is not None:
                frame_for_recog = recognition_frame.copy()

        if frame_for_recog is None:
            time.sleep(0.05)
            continue

        # İlk scriptteki gibi kareyi küçültüp RGB'ye çevir
        small_frame = cv2.resize(frame_for_recog, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

        face_locations = []
        face_names = []

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Bilinmiyor"

                if known_face_encodings:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                        now = time.time()
                        if (name != last_recognized_name) or (now - last_recognized_time > 3.0):
                            last_recognized_name = name
                            last_recognized_time = now
                            if mode == "attendance":
                                log_attendance_hit(name)

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Çizimleri orijinal kare üzerinde yap
        frame_draw = frame_for_recog.copy()
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)
            cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame_draw, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame_draw, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        if not face_names:
            cv2.putText(frame_draw, "Yuz algilanamadi", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        frame_draw_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        with recognition_lock:
            current_frame = frame_draw_rgb.copy()

        time.sleep(0.01)


def update_camera_display():
    global current_frame, camera_label, camera_active
    if camera_label and current_frame is not None:
        try:
            pil_image = Image.fromarray(current_frame)
            tk_image = ImageTk.PhotoImage(pil_image)
            camera_label.configure(image=tk_image)
            camera_label.image = tk_image
        except Exception as e:
            # hata görmezden gel
            print(f"Gosterim hatasi: {e}")
    if camera_active:
        # ~30 FPS ekran güncellemesi (yaklaşık)
        camera_label.after(33, update_camera_display)

def start_camera(mode="capture"):
    global camera_active, camera_thread, camera_label, camera_window
    global recognition_active, recognition_thread

    if camera_active:
        stop_camera()
        time.sleep(0.15)

    camera_window = tk.Toplevel()
    camera_window.title("Kamera")
    # Sabit boyut ver (küçülüp büyüme sorununu çözüyor)
    camera_window.geometry("680x520")
    camera_window.resizable(False, False)

    # Kamera görüntüsü için frame
    top_frame = tk.Frame(camera_window)
    top_frame.pack(fill="both", expand=True, padx=8, pady=8)

    camera_label = tk.Label(top_frame, text="Kamera başlatılıyor...", bg="black", fg="white")
    camera_label.pack(fill="both", expand=True)

    control_frame = tk.Frame(camera_window)
    control_frame.pack(fill="x", padx=10, pady=6)

    if mode == "capture":
        tk.Button(control_frame, text="Fotoğraf Çek", command=capture_photo,
                  font=("Arial", 12, "bold"), width=12, height=1, bg="lightgreen").pack(side=tk.LEFT, padx=6)

    tk.Button(control_frame, text="Kamerayı Kapat", command=stop_camera,
              font=("Arial", 12, "bold"), width=12, height=1, bg="lightcoral").pack(side=tk.LEFT, padx=6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        camera_window.destroy()
        return None

    camera_active = True
    camera_thread = threading.Thread(target=camera_thread_function, args=(cap,), daemon=True)
    camera_thread.start()

    recognition_active = True
    recognition_thread = threading.Thread(target=recognition_worker, args=(mode,), daemon=True)
    recognition_thread.start()

    update_camera_display()

    def on_close():
        stop_camera()
        try:
            camera_window.destroy()
        except: pass
    camera_window.protocol("WM_DELETE_WINDOW", on_close)
    return cap

def stop_camera():
    global camera_active, recognition_active, camera_window
    camera_active = False
    recognition_active = False
    time.sleep(0.1)
    if camera_window:
        try:
            camera_window.destroy()
        except: pass

# ====================== OPENCV TABANLI YOKLAMA ======================
def realtime_attendance_loop():
    global attendance_session_active, last_recognized_name, last_recognized_time, attendance_thread

    ensure_csv_header(YOKLAMA_CSV)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        attendance_session_active = False
        print("Kamera açılamadı!")
        if root and root.winfo_exists():
            root.after(0, lambda: messagebox.showerror("Hata", "Kamera açılamadı!"))
        attendance_thread = None
        return

    cv2.namedWindow(ATTENDANCE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ATTENDANCE_WINDOW, 640, 480)

    process_this_frame = True
    face_locations = []
    face_names = []

    try:
        while attendance_session_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                rgb_small_frame = np.ascontiguousarray(rgb_small_frame)

                detections = face_recognition.face_locations(rgb_small_frame)
                encodings = face_recognition.face_encodings(rgb_small_frame, detections)

                updated_names = []
                for face_encoding in encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Bilinmiyor"

                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            now = time.time()
                            if (name != last_recognized_name) or (now - last_recognized_time > 3.0):
                                last_recognized_name = name
                                last_recognized_time = now
                                log_attendance_hit(name)

                    updated_names.append(name)

                face_locations = detections
                face_names = updated_names

            process_this_frame = not process_this_frame

            frame_draw = frame.copy()
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)
                label = name if name == "Bilinmiyor" else name.replace("_", " ")
                cv2.rectangle(frame_draw, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame_draw, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame_draw, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            if not face_names:
                cv2.putText(frame_draw, "Yuz algilanamadi", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

            cv2.imshow(ATTENDANCE_WINDOW, frame_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            try:
                if cv2.getWindowProperty(ATTENDANCE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        attendance_session_active = False
        cap.release()
        try:
            cv2.destroyWindow(ATTENDANCE_WINDOW)
        except cv2.error:
            pass
        attendance_thread = None


def start_attendance_session():
    global attendance_session_active, attendance_thread
    if attendance_session_active:
        messagebox.showinfo("Yoklama", "Yoklama penceresi zaten açık. Çıkmak için 'q' tuşuna basın.")
        return False

    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        test_cap.release()
        messagebox.showerror("Hata", "Kamera açılamadı!")
        return False
    test_cap.release()

    attendance_session_active = True
    attendance_thread = threading.Thread(target=realtime_attendance_loop, daemon=True)
    attendance_thread.start()
    return True


def stop_attendance_session():
    global attendance_session_active, attendance_thread
    if attendance_session_active:
        attendance_session_active = False
    if attendance_thread and attendance_thread.is_alive():
        attendance_thread.join(timeout=1.0)
    attendance_thread = None

# ====================== FOTOĞRAF İŞLEMLERİ ======================
def capture_photo():
    global captured_frame, current_frame
    if current_frame is not None:
        captured_frame = current_frame.copy()
        messagebox.showinfo("Başarılı", "Fotoğraf çekildi!")
        return captured_frame
    return None

def process_captured_photo(ad, soyad):
    global captured_frame
    if captured_frame is None:
        messagebox.showerror("Hata", "Fotoğraf yok.")
        return
    frame_bgr = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
    safe_ad = ad.strip().replace(" ", "")
    safe_soyad = soyad.strip().replace(" ", "")
    foto_yolu = FOTO_KLASOR / f"{safe_ad}_{safe_soyad}.jpg"
    cv2.imwrite(str(foto_yolu), frame_bgr)
    try:
        if cursor and db:
            cursor.execute("UPDATE ogrenciler SET foto_yolu=%s WHERE ad=%s AND soyad=%s", (str(foto_yolu), ad, soyad))
            db.commit()
            model_egit_ve_kaydet()
            messagebox.showinfo("Başarılı", f"{ad} {soyad} eklendi.")
        else:
            messagebox.showwarning("Uyarı", "Sadece fotoğraf kaydedildi.")
    except Exception as e:
        messagebox.showwarning("Uyarı", f"Hata: {e}")
    captured_frame = None

# ====================== ÖĞRENCİ İŞLEMLERİ ======================
def yeni_ogrenci_kaydet():
    if not cursor or not db:
        messagebox.showerror("Hata", "Veritabanı yok!")
        return
    ad = simpledialog.askstring("Yeni Öğrenci", "Ad:")
    soyad = simpledialog.askstring("Yeni Öğrenci", "Soyad:")
    if not ad or not soyad: return

    gun_window = tk.Toplevel(root)
    gun_window.title("Ders Günleri")
    tk.Label(gun_window, text="Ders günlerini seç (Ctrl ile birden fazla)").pack(pady=5)
    listbox = tk.Listbox(gun_window, selectmode=tk.MULTIPLE)
    for gun in ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]:
        listbox.insert(tk.END, gun)
    listbox.pack(padx=10, pady=10)

    def kaydet():
        secilen = [listbox.get(i) for i in listbox.curselection()]
        if not secilen:
            messagebox.showerror("Hata", "Gün seçin!")
            return
        gun_str = ",".join(secilen)
        gun_window.destroy()
        messagebox.showinfo("Kamera", "Fotoğraf çekilecek...")
        cap = start_camera("capture")
        if not cap: return

        def check():
            if camera_window and camera_window.winfo_exists():
                root.after(200, check)
            else:
                if captured_frame is not None:
                    cursor.execute("INSERT INTO ogrenciler (ad, soyad, gun) VALUES (%s,%s,%s)", (ad, soyad, gun_str))
                    db.commit()
                    process_captured_photo(ad, soyad)
        root.after(200, check)
    tk.Button(gun_window, text="Kaydet", command=kaydet).pack(pady=5)

def yoklama_al():
    model_yukle()
    if not known_face_encodings:
        messagebox.showerror("Hata", "Öğrenci ekleyin.")
        return
    if start_attendance_session():
        messagebox.showinfo(
            "Yoklama",
            f"YEŞİL = Kayıtlı\nKIRMIZI = Bilinmiyor\n{len(known_face_names)} öğrenci\n"
            "Pencereyi kapatmak için 'q' tuşuna basın."
        )

# ====================== RAPORLAMA ======================
def yoklamayi_excele_aktar():
    if not cursor: return
    cursor.execute("SELECT o.ad, o.soyad, y.tarih, y.durum FROM yoklama y JOIN ogrenciler o ON o.id = y.ogrenci_id ORDER BY y.tarih")
    rows = cursor.fetchall()
    if not rows:
        messagebox.showinfo("Bilgi", "Kayıt yok.")
        return
    df = pd.DataFrame(rows, columns=["Ad", "Soyad", "Tarih", "Durum"])
    dosya = f"yoklama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(dosya, index=False)
    messagebox.showinfo("Excel", f"Kaydedildi: {dosya}")

def yoklama_sayac_goster():
    if not cursor: return
    cursor.execute("SELECT o.ad, o.soyad, COUNT(y.id) FROM ogrenciler o LEFT JOIN yoklama y ON o.id = y.ogrenci_id GROUP BY o.id")
    rows = cursor.fetchall()
    win = tk.Toplevel(root)
    win.title("Yoklama Sayacı")
    text = tk.Text(win)
    for ad, soyad, sayi in rows:
        text.insert(tk.END, f"{ad} {soyad}: {sayi} gün\n")
    text.pack()

def kayitli_ogrencileri_goster():
    if not cursor: return
    cursor.execute("SELECT ad, soyad, gun FROM ogrenciler")
    rows = cursor.fetchall()
    win = tk.Toplevel(root)
    win.title("Kayıtlı Öğrenciler")
    text = tk.Text(win)
    for ad, soyad, gun in rows:
        text.insert(tk.END, f"{ad} {soyad} - {gun}\n")
    text.pack()

def delete_student():
    if not cursor: return
    ogr_id = simpledialog.askinteger("Sil", "Öğrenci ID:")
    if not ogr_id: return
    cursor.execute("SELECT ad, soyad, foto_yolu FROM ogrenciler WHERE id=%s", (ogr_id,))
    row = cursor.fetchone()
    if not row: 
        messagebox.showinfo("Bilgi", "ID yok.")
        return
    ad, soyad, foto = row
    cursor.execute("DELETE FROM yoklama WHERE ogrenci_id=%s", (ogr_id,))
    cursor.execute("DELETE FROM ogrenciler WHERE id=%s", (ogr_id,))
    db.commit()
    if foto and os.path.exists(foto):
        os.remove(foto)
    messagebox.showinfo("Başarılı", f"{ad} {soyad} silindi.")
    model_egit_ve_kaydet()

# ====================== ANA ARAYÜZ ======================
def main():
    global root
    root = tk.Tk()
    root.title("Yüz Tanıma Yoklama Sistemi")
    root.geometry("420x400")
    root.resizable(False, False)
    style = {"width": 30, "height": 1, "pady": 6}

    def on_close():
        stop_attendance_session()
        stop_camera()
        if db: db.close()
        root.destroy()

    tk.Button(root, text="Yeni Öğrenci Ekle", command=yeni_ogrenci_kaydet, **style).pack()
    tk.Button(root, text="Yoklama Al", command=yoklama_al, **style).pack()
    tk.Button(root, text="Yoklama Sayacı", command=yoklama_sayac_goster, **style).pack()
    tk.Button(root, text="Excel'e Aktar", command=yoklamayi_excele_aktar, **style).pack()
    tk.Button(root, text="Kayıtlı Öğrenciler", command=kayitli_ogrencileri_goster, **style).pack()
    tk.Button(root, text="Öğrenci Sil", command=delete_student, **style).pack()
    tk.Button(root, text="Çıkış", command=on_close, **style).pack()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        model_yukle()
    except Exception as e:
        print(f"Model hatası: {e}")

    root.mainloop()

if __name__ == "__main__":
    main()
    
