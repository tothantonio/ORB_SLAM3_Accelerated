import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATELE TALE DE PROFILARE (Timp Mediu in ms) ---
data = {
    'Componenta': [
        'ORB Extraction', 'Pose Prediction', 'LM Track', 'New KF decision',
        'KF Insertion', 'MP Culling', 'MP Creation', 'LBA', 'KF Culling'
    ],
    'Timp_Mediu_ms': [
        26.11931, 2.43939, 4.27669, 0.15352,
        10.48579, 0.20646, 53.21622, 114.69051, 9.12434
    ],
    'Thread': [
        'Tracking', 'Tracking', 'Tracking', 'Tracking',
        'Local Mapping', 'Local Mapping', 'Local Mapping', 'Local Mapping', 'Local Mapping'
    ]
}

df = pd.DataFrame(data)

# Totalurile Thread-urilor (pentru calcul procentual)
tracking_total = 35.22639
mapping_total = 186.57781

# --- 2. GRAFIC 1: Top 5 Blocaje (Timp Mediu Absolut) ---

# Excludem componentele mici din Tracking pentru a ne concentra pe LBA/MP Creation (cele mai mari)
top_abs = df.sort_values(by='Timp_Mediu_ms', ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.barh(top_abs['Componenta'], top_abs['Timp_Mediu_ms'], color=['#e67e22', '#2980b9', '#34495e', '#2980b9', '#2980b9'])
plt.xlabel('Timp Mediu (ms)')
plt.title('Top 5 Candidați la Accelerare (Timp Mediu Absolut)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('grafic_blocaje_absolute.png')
print("Graficul 'grafic_blocaje_absolute.png' a fost generat.")


# --- 3. GRAFIC 2: Procente (Pie Charts) ---

# a) Date Tracking (Front-end)
tracking_data = df[df['Thread'] == 'Tracking']
tracking_data['Procent'] = (tracking_data['Timp_Mediu_ms'] / tracking_total) * 100

# b) Date Local Mapping (Back-end)
mapping_data = df[df['Thread'] == 'Local Mapping']
mapping_data['Procent'] = (mapping_data['Timp_Mediu_ms'] / mapping_total) * 100


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
colors_tracking = ['#3498db', '#f39c12', '#9b59b6', '#bdc3c7'] # Culori pentru Tracking
colors_mapping = ['#27ae60', '#e74c3c', '#f1c40f', '#3498db', '#95a5a6'] # Culori pentru Mapping

# Pie Chart 1: Tracking
ax1.pie(
    tracking_data['Procent'],
    labels=[f"{c} ({p:.1f}%)" for c, p in zip(tracking_data['Componenta'], tracking_data['Procent'])],
    autopct='',
    startangle=90,
    colors=colors_tracking
)
ax1.set_title(f'Contribuția Tracking (Total: {tracking_total:.2f} ms)')

# Pie Chart 2: Local Mapping
ax2.pie(
    mapping_data['Procent'],
    labels=[f"{c} ({p:.1f}%)" for c, p in zip(mapping_data['Componenta'], mapping_data['Procent'])],
    autopct='',
    startangle=90,
    colors=colors_mapping
)
ax2.set_title(f'Contribuția Local Mapping (Total: {mapping_total:.2f} ms)')

plt.suptitle('Analiza Procentuală a Timpului de Execuție pe Thread-uri')
fig.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('grafic_procentual_threads.png')
print("Graficul 'grafic_procentual_threads.png' a fost generat.")