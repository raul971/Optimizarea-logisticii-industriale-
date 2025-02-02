# Demo BFS + Q-table + Deep Q + Neural D* + Barcode

Acesta este un proiect demonstrativ care combină algoritmi de rutare, scanare de coduri de bare și o interfață grafică interactivă. 

## Caracteristici principale

- **Algoritmi de rutare:** 
  - BFS (Breadth-First Search)
  - Q-learning tabular
  - Neural D* (D* cu euristică bazată pe rețele neuronale)
  - Placeholder pentru Deep Q Learning
- **Scanare coduri de bare:** 
  - Folosind `pyzbar` pentru decodare reală
  - Model neuronal simplificat pentru recunoaștere
- **Interfață grafică (`tkinter`)** pentru setarea hărții, punctelor de start și a depozitelor
- **Server Flask** pentru compararea rutelor între algoritmi

---

## Cerințe

- Python 3.x
- Biblioteci necesare:
  ```bash
  pip install opencv-python numpy matplotlib pyzbar pillow tensorflow flask
  ```

*(Dacă ai placă grafică NVIDIA, poți instala `tensorflow-gpu` pentru performanță mai bună.)*

---

## Cum Rulezi Aplicația

1. Clonează sau descarcă repository-ul:
   ```bash
   git clone https://github.com/user/repo.git
   cd repo
   ```
2. Instalează dependențele (vezi secțiunea [Cerințe](#cerințe)).
3. Rulează aplicația:
   ```bash
   python main.py
   ```
4. Se vor deschide:
   - **Interfața grafică Tkinter** pentru gestionarea rutei
   - **Serverul Flask** accesibil la `http://127.0.0.1:5000/`

---

## Interfața Grafică (Tkinter)

- **Canvas hartă:** Poți adăuga obstacole, seta puncte de start și depozite.
- **Butoane:**
  - `Start Camera` – pornește captura webcam pentru scanare coduri de bare
  - `Fake Barcode` – simulează un cod de bare pentru testare
  - `Save Layout`, `Load Layout`, `Reset Layout` – pentru salvarea configurației
- **Selectare algoritm de rutare:** BFS, Q-table, Deep Q, Neural D*
- **LED status robot:**
  - 🔴 Roșu: inactiv
  - 🟡 Galben: încărcare
  - 🟢 Verde: în mișcare

---

## Server Flask

- **Endpoint principal:** `http://127.0.0.1:5000/compare-route`
- **Exemplu de utilizare:**
  ```
  http://127.0.0.1:5000/compare-route?method2=q_table&start_x=0&start_y=0&end_x=9&end_y=9
  ```
- **Ce face:**
  - Compară BFS vs o altă metodă (Q-table, Neural D*, Deep Q)
  - Generează o imagine cu traseele și evoluția costului/recompenselor

---

## Structura Codului

- **`main.py`** – cod principal, gestionează interfața Tkinter și serverul Flask
- **`FactoryMap`** – gestionează harta și obstacolele
- **`RobotSimulator`** – logica de transport și livrare
- **`VisionSystem`** – scanare și identificare piese
- **`compute_route_*`** – implementările algoritmilor de rutare

---

## Observații

- **Camera trebuie să fie disponibilă** dacă vrei să testezi scanarea codurilor de bare.
- **Rutarea Neural D*** folosește un model Keras antrenat rapid – poate fi îmbunătățit cu date reale.
- **Deep Q este momentan un placeholder** și trebuie implementat complet pentru o rețea neuronală funcțională.

Sperăm că acest proiect îți va fi util! 🚀
