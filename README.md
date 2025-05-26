
# 3D-Engine in Python

Eine einfache 3D-Engine in Python, die .obj-Modelle lädt, transformiert und perspektivisch rendert. Sie basiert auf Pygame und NumPy und unterstützt grundlegende Kamerafunktionen, Beleuchtung per Dot-Produkt, Backface Culling und Tiefensortierung. Ein zufällig generierter Sternenhintergrund ergänzt die Szene visuell.
Das Projekt dient als erster praktischer Einstieg in Grafikprogrammierung und enthält entsprechend noch einige technische Ungenauigkeiten, Limitierungen und Bugs. 

### Raumschiff Render in 3d-PyEngine

https://github.com/user-attachments/assets/a11cdcab-f939-4f26-b7ee-d837f8974eed





# Features

### 3D-Rendering
- 3D Rendering Pipeline
  - Vertices → Transformation → Projektion → Clipping → Rasterisierung
- Perspektivische Projektion
  - Perspektivische Projektion mit anpassbarem Field of View (FOV), Nah- und Fern-Ebene (znear, zfar)
- Clipping
  - Flächen-Clipping gegen Bildschirmgrenzen (Normalized Device Coordinates - NDC)
  - Basic Clipping (Grundlegendes Clipping)
- Beleuchtung
  - Skalarprodukt-Beleuchtung aus einer Richtung (sehr vereinfacht)
  - Einfache Lichtberechnung (Dot-Produkt Normal × Licht-Richtung)
  - Dynamische Beleuchtung mit Helligkeitsanpassung pro Polygon 
- Backface Culling
  - Zeichnen nur der sichtbaren Flächen basierend auf Normalvektoren
- Tiefensortierung
  - Z-Buffer oder Painter's Algorithm zur Reihenfolge der Darstellung
- Zeichnen von Polygonen
  - Flächen mit Füllung und/oder Kantenkontur
- Rendering von mehreren Objekten
  - Mehrere Objekte mit unabhängiger Position, Rotation und Skalierung

### 3D-Objektmanagement
- Laden von 3D-Modellen aus .obj-Dateien
- Objekt-Transformationen
  - Translation, Rotation (X, Y, Z) und Skalierung von Objekten
- Sehr simple Animation von Objekten
  - Objektrotation und Positionsänderungen im Programm-Loop
- Beispiel-Objekte:
  - VideoShip, Teekanne, Achsen, Planeten, Energie-Nodes

### Kamera- und Steuerungs-Features
- Kamerasteuerung
  - Bewegung der Kamera mit Pfeiltasten (Position verändern)
  - Rotation der Kamera um Y-Achse (Yaw) möglich
  - Verknüpfung von Kamera-Position und Blickrichtung (lookAt-Funktion)
  - Free-Look Steuerung (Rotationen via Yaw)
- FPS-gesteuerte Animation und Objektbewegungen

### Grafik und Darstellung
- Sternenhimmel als Background mit zufälligen kleinen Sternenpunkten
- Clipping von Flächen an Bildschirmrändern für sauberes Rendering
- Antialiasing-Effekt durch Kantenschattierung (Polygon-Konturen)

# Probleme und Herausforderungen

- Performance und Ressourcen
  - Python ist nicht sehr rechenstark, viele gleichzeitige Rechnungen sind schwierig wegen fehlendem Memory Management
  - Rendering der Teekanne (Hello World des Graphic Programmings) nicht möglich wegen zu vieler Vertices
- Zeitdruck
  - Projekt musste innerhalb von 2 Wochen fertiggestellt werden (Abiturzeit)
  - Fokus lag mehr auf Verstehen und Lernen als auf perfektes Programmieren
- Erfahrung und Projektmanagement
  - Erstes Python-Projekt, daher mangelnde Erfahrung im richtigen Management von Python-Projekten
- Bugs und Fehlersuche
  - Viele Bugs im Programm
  - Schwierigkeit beim Bug Fixing wegen mangelndem Verständnis der zugrundeliegenden Mathematik
- Sprachliche und fachliche Hürden
  - Viele Lernvideos und Materialien auf Englisch
  - Fachbegriffe wie Skalarprodukt, Dot-Product waren zunächst unverständlich

## Zukunft
Ich hoffe, dieses Projekt in Zukunft erneut aufzugreifen, jedoch in einer speichereffizienteren Sprache wie C++. Dort eröffnen sich zahlreiche Möglichkeiten, das Projekt deutlich zu erweitern. Denkbar sind zum Beispiel komplexe Animationen, das Einbinden von Texturen, fortgeschrittene Beleuchtungstechniken und Optimierungen der Rendering-Pipeline.

Insgesamt war dieses Projekt ein sehr guter Einstieg und hat mir geholfen, die grundlegenden Konzepte des 3D-Renderings und der Computergrafik zu verstehen. Mit mehr Zeit und Erfahrung möchte ich diese Basis nutzen, um ein noch anspruchsvolleres und leistungsfähigeres Grafikprogramm zu entwickeln.
