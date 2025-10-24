"""
PASO 3: Entrenar clasificador de poses con keypoints de Kinetics-700
Incluye manejo de desbalance de clases y métricas detalladas
"""
import json
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

print("="*70)
print("PASO 3: ENTRENAMIENTO DEL CLASIFICADOR")
print("="*70)

# ==========================================
# CONFIGURACIÓN
# ==========================================

INPUT_FILE = "data/pose_dataset_kinetics.json"
MODEL_FILE = "data/pose_classifier.pkl"

print(f"\n📋 Configuración:")
print(f"   Archivo de entrada: {INPUT_FILE}")
print(f"   Modelo de salida: {MODEL_FILE}")

# ==========================================
# CARGAR DATOS
# ==========================================

print(f"\n📂 Cargando dataset...")
try:
    with open(INPUT_FILE, 'r') as f:
        dataset = json.load(f)
    print(f"   ✅ Dataset cargado: {len(dataset)} samples")
except FileNotFoundError:
    print(f"   ❌ Error: No se encontró {INPUT_FILE}")
    print(f"   Ejecuta primero: python 2_extract_keypoints.py")
    exit(1)

# ==========================================
# PREPARAR DATOS
# ==========================================

print(f"\n🔧 Preparando datos...")

# Primero inspeccionar formato del primer sample
print(f"\n🔍 Inspeccionando formato de datos...")
sample_0 = dataset[0]
print(f"   Claves del sample: {sample_0.keys()}")
print(f"   Tipo de keypoints: {type(sample_0['keypoints'])}")
print(f"   Primer keypoint: {sample_0['keypoints'][0]}")
print(f"   Formato detectado: {'dict' if isinstance(sample_0['keypoints'][0], dict) else 'list'}")

X = []  # Features (keypoints)
y = []  # Labels (clases)

for sample in dataset:
    keypoints = sample['keypoints']
    label = sample['label']
    
    # Aplanar keypoints adaptándose al formato
    flattened = []
    for kp in keypoints:
        # Si es diccionario: {'x': valor, 'y': valor}
        if isinstance(kp, dict):
            flattened.extend([kp['x'], kp['y']])
        # Si es lista: [x, y] o [x, y, conf]
        elif isinstance(kp, (list, tuple)):
            flattened.extend([kp[0], kp[1]])
        else:
            print(f"   ⚠️ Formato inesperado en keypoint: {type(kp)}")
    
    X.append(flattened)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"   Shape de X: {X.shape}")
print(f"   Shape de y: {y.shape}")

# Mostrar distribución de clases
class_counts = Counter(y)
print(f"\n📊 Distribución de clases:")
for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(y)) * 100
    print(f"   {cls:20s}: {count:5d} samples ({percentage:5.1f}%)")

# ==========================================
# SPLIT TRAIN/TEST
# ==========================================

print(f"\n✂️ Dividiendo en train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantener proporciones en train y test
)

print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples")

# ==========================================
# CALCULAR PESOS DE CLASES (para desbalance)
# ==========================================

print(f"\n⚖️ Calculando pesos para balancear clases...")
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

print(f"   Pesos por clase:")
for cls in sorted(class_weight_dict.keys()):
    print(f"   {cls:20s}: {class_weight_dict[cls]:.3f}")

# ==========================================
# ENTRENAR MODELO
# ==========================================

print(f"\n🤖 Entrenando Random Forest Classifier...")
print(f"   (esto puede tomar 2-5 minutos...)")

clf = RandomForestClassifier(
    n_estimators=200,           # Número de árboles
    max_depth=20,               # Profundidad máxima
    min_samples_split=5,        # Mínimo de samples para split
    min_samples_leaf=2,         # Mínimo de samples en hoja
    class_weight=class_weight_dict,  # Balancear clases
    random_state=42,
    n_jobs=-1,                  # Usar todos los cores
    verbose=1
)

clf.fit(X_train, y_train)
print(f"   ✅ Modelo entrenado")

# ==========================================
# VALIDACIÓN CRUZADA
# ==========================================

print(f"\n🔄 Ejecutando validación cruzada (5-fold)...")
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"   Scores por fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"   Promedio: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ==========================================
# EVALUAR EN TEST
# ==========================================

print(f"\n📈 Evaluando en conjunto de test...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print(f"🎯 ACCURACY EN TEST: {accuracy:.1%}")
print(f"{'='*70}")

# ==========================================
# REPORTE DETALLADO
# ==========================================

print(f"\n📊 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred, digits=3))

# ==========================================
# MATRIZ DE CONFUSIÓN
# ==========================================

print(f"\n📉 Generando matriz de confusión...")
cm = confusion_matrix(y_test, y_pred, labels=sorted(classes))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=sorted(classes),
    yticklabels=sorted(classes)
)
plt.title(f'Matriz de Confusión\nAccuracy: {accuracy:.1%}')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()

confusion_matrix_file = 'data/confusion_matrix.png'
plt.savefig(confusion_matrix_file, dpi=150, bbox_inches='tight')
print(f"   ✅ Guardada en: {confusion_matrix_file}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print(f"\n🔍 Analizando importancia de features...")
feature_importance = clf.feature_importances_

# Agrupar por keypoint (cada keypoint tiene x,y)
keypoint_names = [
    'nariz', 'ojo_izq', 'ojo_der', 'oreja_izq', 'oreja_der',
    'hombro_izq', 'hombro_der', 'codo_izq', 'codo_der',
    'muñeca_izq', 'muñeca_der', 'cadera_izq', 'cadera_der',
    'rodilla_izq', 'rodilla_der', 'tobillo_izq', 'tobillo_der'
]

keypoint_importance = []
for i in range(17):
    importance = feature_importance[i*2] + feature_importance[i*2+1]
    keypoint_importance.append((keypoint_names[i], importance))

keypoint_importance.sort(key=lambda x: x[1], reverse=True)

print(f"\n   Top 5 keypoints más importantes:")
for kp_name, importance in keypoint_importance[:5]:
    print(f"   {kp_name:15s}: {importance:.4f}")

# ==========================================
# GUARDAR MODELO
# ==========================================

print(f"\n💾 Guardando modelo...")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)
print(f"   ✅ Modelo guardado en: {MODEL_FILE}")

# ==========================================
# RESUMEN FINAL
# ==========================================

print(f"\n{'='*70}")
print(f"✅ ENTRENAMIENTO COMPLETADO")
print(f"{'='*70}")
print(f"\n📊 Resumen:")
print(f"   Total samples:        {len(dataset):,}")
print(f"   Train samples:        {len(X_train):,}")
print(f"   Test samples:         {len(X_test):,}")
print(f"   Accuracy en test:     {accuracy:.1%}")
print(f"   Cross-validation:     {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print(f"\n📁 Archivos generados:")
print(f"   Modelo:               {MODEL_FILE}")
print(f"   Matriz de confusión:  {confusion_matrix_file}")
print(f"\n🎯 SIGUIENTE PASO:")
print(f"   Integrar el modelo en tu sistema de detección")
print(f"   python main.py")
print()