#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_face_recognition
----------------------------------

Tests for `face_recognition` module.
"""

#WEB library
import streamlit.components.v1 as components
import streamlit as st

import unittest
import os
from click.testing import CliRunner

from face_recognition import api
from face_recognition import face_recognition_cli
from face_recognition import face_detection_cli

#opencv library
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import time


FRAME_WINDOW = st.image([]) #frame window

hhide_st_style = """ 
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hhide_st_style, unsafe_allow_html=True) #hide streamlit menu

menu = ["HOME", "DETEKSI KEHADIRAN", "REGISTER DATA", "DAFTAR PRESENSI", "HAPUS DATA"] #menu

#sidebar
st.sidebar.image('static/images/IN.png', width=100)
st.sidebar.title("faceIN")
choice = st.sidebar.selectbox("Menu", menu) #sidebar menu

path = 'absensi' #path to save image
images = [] #list of image
classNames = [] #list of class
myList = os.listdir(path) #list of image
for cl in myList: #loop
    classNames.append(os.path.splitext(cl)[0]) #split image name

col1, col2, col3 = st.columns(3) #columns
cap = cv2.VideoCapture(0) #capture video

if choice == 'DETEKSI KEHADIRAN': 
    st.markdown("<h2 style='text-align: center; color: white;'>DETEKSI KEHADIRAN</h2>", unsafe_allow_html=True)
    run = st.checkbox("Jalankan kamera") #checkbox
    if run == True:
        for cl in myList: #loop
            curlImg = cv2.imread(f'{path}/{cl}') #read image
            images.append(curlImg)
        print(classNames)

        def findEncodings(images): #find encoding
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    encode = encodings[0]
                    encodeList.append(encode)
                else:
                    print(f"wajah tidak ditemukan {img}")
            return encodeList

        def faceList(name):
            with open('absensi.csv', 'r+') as f:
                myDataList = f.readlines()
                dateToday = datetime.now().strftime('%d-%m-%Y')
                for line in myDataList:
                    entry = line.split(',')
                    if len(entry) >= 3:
                        entryName = entry[0]
                        entryDate = entry[2].strip()
                        if entryName == name and entryDate == dateToday:
                            return  # berhenti menambahkan data jika sudah hadir
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString},{dateToday}')

        encodeListUnknown = findEncodings(images)
        print('encoding complete!')

        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListUnknown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListUnknown, encodeFace)
                #print(faceDis)
                matchesIndex = np.argmin(faceDis)
                
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                if matches[matchesIndex]:
                    name = classNames[matchesIndex].upper()
                    print(name)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    faceList(name)

                    time.sleep(3)
                
                else:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            FRAME_WINDOW.image(img)
            cv2.waitKey(1)
    else:
        pass

# Register menu
elif choice == 'REGISTER DATA':
    st.markdown("<h2 style='text-align: center; color: white;'>REGISTER DATA</h2>", unsafe_allow_html=True)
        
    # Function to load image from bytes
    def load_image(image_file):
        img = Image.open(image_file)
        return img

    # Function to capture image from camera
    def capture_image():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame

    # Input for name
    user_name = st.text_input("Masukkan nama:")
    
    st.write("Klik tombol dibawah ini untuk mengambil gambar sebagai data wajah")
    capture_btn = st.button("Mengambil Gambar")
    
    if capture_btn:
        if user_name == "":
            st.warning("Tolong masukkan nama sebelum mengambil gambar.")
        else:
            img = capture_image()
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                st.image(img_pil, caption="Captured Image")
                
                # Save the captured image
                if not os.path.exists("absensi"):
                    os.makedirs("absensi")
                
                img_pil.save(os.path.join("absensi", f"{user_name}.png"))
                st.success(f"Saved Captured Image as {user_name}.png")

#daftar presensi menu
elif choice == 'DAFTAR PRESENSI':
    st.markdown("<h2 style='text-align: center; color: white;'>DAFTAR KEHADIRAN</h2>", unsafe_allow_html=True)
    if os.path.exists('absensi.csv'):
        df = pd.read_csv('absensi.csv', header=None, names=['NAMA', 'WAKTU HADIR', 'TANGGAL'])
        df = df.dropna(subset=['TANGGAL'])  # Drop rows with missing dates
        unique_dates = df['TANGGAL'].unique()
        for date in unique_dates:
            st.write(f"### Daftar Kehadiran Tanggal {date}")
            date_df = df[df['TANGGAL'] == date]
            st.write(date_df)
    else:
        st.write("Data tidak ditemukan")

#home menu        
elif choice == 'HOME':
    st.title("Selamat datang di faceIN!")
    
    st.write("""
    faceIN adalah sebuah sistem inovatif yang dirancang untuk memudahkan para pengajar dalam mengambil kehadiran siswa atau mahasiswanya. Menggunakan teknologi face recognition yang canggih, faceIN mampu mengenali wajah dengan akurat dan mencatat data kehadiran secara otomatis dan ada waktu kehadirannya.
    """)

    st.subheader("Fitur:")
    
    st.write("""
    - **Pendeteksian Wajah yang Akurat:** Dengan algoritma face recognition yang canggih, faceIN dapat mengenali wajah setiap siswa atau mahasiswa dengan tingkat akurasi yang tinggi.
    - **Pencatatan Kehadiran Otomatis:** Data kehadiran siswa atau mahasiswa dicatat secara otomatis dan real-time waktu kehadirannya.
    """)

    st.write("Dengan faceIN proses absensi menjadi lebih efisien dan akurat, sehingga Anda dapat fokus pada pengajaran dan pembelajaran yang lebih baik. Selamat menggunakan faceIN!")


elif choice == "HAPUS DATA":
    st.markdown("<h2 style='text-align: center; color: white;'>HAPUS DATA</h2>", unsafe_allow_html=True)
    
    # Function to delete a face data file
    def delete_face_data(file_name):
        file_path = os.path.join("absensi", file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        else:
            return False
    
    # Display the list of registered names with delete options
    if len(myList) > 0:
        selected_name = st.selectbox("Pilih nama untuk dihapus datanya", classNames)
        delete_btn = st.button("Hapus Data Wajah")
        
        if delete_btn:
            file_name = f"{selected_name}.png"
            success = delete_face_data(file_name)
            if success:
                st.success(f"Data wajah untuk {selected_name} berhasil dihapus.")
            else:
                st.error(f"Data wajah untuk {selected_name} tidak ditemukan.")
    else:
        st.write("Tidak ada data wajah yang terdaftar.")
