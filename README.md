# Detector de Imágenes Generadas por IA

Este repositorio contiene un proyecto de detección de imágenes generadas por IA. Desarrollamos un modelo personalizado y lo comparamos con soluciones existentes, mejorando varios modelos mediante fine-tuning con nuestro conjunto de datos.

## Índice
1. [Introducción](#introducción)
2. [Requisitos e Instalación](#requisitos-e-instalación)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Modelos](#modelos)
   - [Nuestro Modelo](#nuestro-modelo)
   - [Nuestro Modelo Fine-tuned](#nuestro-modelo-fine-tuned)
   - [CNN Detection](#cnn-detection)
   - [FaceForensics](#faceforensics)
   - [FaceForensics Fine-tuned](#faceforensics-fine-tuned)
5. [Comparación de modelos](#comparación-de-modelos)
6. [Interfaz para análisis de imágenes](#interfaz-para-análisis-de-imágenes)
7. [Referencias](#referencias)

## Introducción

Con el aumento de imágenes generadas por IA, la capacidad de detectar automáticamente si una imagen es real o sintética se ha vuelto crucial. Este proyecto explora diferentes enfoques para la detección de imágenes generadas por IA, desarrollando y comparando varios modelos.

Nuestros objetivos principales fueron:
- Desarrollar un modelo propio para detectar imágenes generadas por IA
- Comparar nuestro modelo con soluciones existentes
- Mejorar tanto nuestro modelo como modelos existentes mediante fine-tuning
- Proporcionar una interfaz para analizar imágenes con todos los modelos

## Requisitos e Instalación

```bash
# Clonar el repositorio
git clone https://github.com/hectorParamio/DetectandoLoFalsoEnUnMundoGeneradoPorIA.git
cd DetectandoLoFalsoEnUnMundoGeneradoPorIA

# Crear entorno virtual (recomendado)
python -m venv venv

venv\Scripts\activate  # En Linux: source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye todas las bibliotecas necesarias para ejecutar el proyecto, incluyendo:
- PyTorch y Torchvision para el desarrollo de modelos
- OpenCV y Pillow para procesamiento de imágenes
- Matplotlib para visualización
- Scikit-learn para métricas y evaluación
- KaggleHub para descarga de datasets
- Face-alignment para detección y alineación de rostros

**Nota importante sobre la estructura de archivos:** Para que los modelos funcionen correctamente, debes asegurarte de que los archivos estén organizados según la siguiente estructura:

1. **Datasets**:
   - Ruta principal: `datasets/image/AI-Face-Detection/`
   - Imágenes reales: `datasets/image/AI-Face-Detection/real/`
   - Imágenes falsas/generadas por IA: `datasets/image/AI-Face-Detection/fake/`

2. **Modelos pre-entrenados originales**:
   - CNN Detection: `models/cnndetection_image/weights/blur_jpg_prob0.5.pth`
   - FaceForensics: Se descarga automáticamente desde la URL definida en el código
   - Componente de FaceForensics (detector Blazeface): `models/faceforensics_image/blazeface/blazeface.pth`
   - Nuestro modelo personalizado: `ourmodel_results/ours.pth`

3. **Modelos fine-tuned**:
   - FaceForensics fine-tuned: `faceforensics_finetunned_results/faceforensics_finetuned.pth`
   - Nuestro modelo fine-tuned: `ourmodel_finetunned_results/ours_finetunned.pth`

4. **Directorios de resultados** (creados automáticamente por los scripts):
   - Modelo original:
     - `ourmodel_results/`
     - `faceforensics_results/`
     - `cnndetection_results/`
   - Modelo fine-tuned:
     - `ourmodel_finetunned_results/`
     - `faceforensics_finetunned_results/`

Los directorios de resultados se crearán automáticamente al ejecutar los scripts.

**Importante:** Para poder ejecutar el fine-tuning (scripts `models_ours_finetunned.py` y `model_faceforensics_finetuned.py`), es necesario que los modelos originales correspondientes ya estén descargados y ubicados correctamente en las rutas especificadas.

### Descarga directa de modelos pre-entrenados

Debido al tamaño de los archivos de modelos pre-entrenados, estos no están incluidos directamente en el reposirtorio. Para facilitar su acceso, hemos subido todos los modelos necesarios a Google Drive con la estructura de carpetas ya organizada. También puedes encontrar una carpeta con los datasets descargados directamente siguiendo la estructura deseada.:

[Enlace a modelos pre-entrenados y dataset de imágenes en Google Drive](https://drive.google.com/drive/folders/12ZETTNHKC1LPfiE26xyK_mpIs90Chlv2?usp=sharing)

Si tienes problemas para configurar la estructura de carpetas o necesitas acceder rápidamente a cualquiera de los modelos, puedes descargarlos directamente desde este enlace y colocarlos en sus respectivas carpetas en el proyecto.

## Estructura del proyecto

- `model_ours.py`: Nuestro modelo CNN para detección de imágenes generadas por IA
- `models_ours_finetunned.py`: Versión fine-tuned de nuestro modelo
- `model_cnndetection.py`: Implementación del modelo CNN Detection
- `model_faceforensics.py`: Implementación del modelo FaceForensics
- `model_faceforensics_finetuned.py`: Versión fine-tuned del modelo FaceForensics
- `compare_models.py`: Script para comparar el rendimiento de todos los modelos
- `interface_models.py`: Interfaz gráfica para analizar imágenes con todos los modelos

## Datasets utilizados

Para el entrenamiento y fine-tuning de nuestros modelos, utilizamos los siguientes conjuntos de datos:

1. [Detect AI Generated Faces - High Quality Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset): El conjunto de datos contiene unas 3.000 imágenes de rostros humanos, tanto reales como generadas por IA.

2. [Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset): El conjunto de datos contiene unas 9.600 imágenes de rostros humanos, tanto reales como generadas por IA.

### Estructura de carpetas para los datasets
Para que los modelos funcionen correctamente, es esencial que organices tus imágenes siguiendo esta estructura específica:

```
datasets/
└── image/
    └── AI-Face-Detection/
        ├── real/
        │   ├── imagen_real_1.jpg
        │   ├── imagen_real_2.jpg
        │   └── ...
        └── fake/
            ├── imagen_falsa_1.jpg
            ├── imagen_falsa_2.jpg
            └── ...
```

- La carpeta `real` debe contener exclusivamente imágenes de rostros reales
- La carpeta `fake` debe contener exclusivamente imágenes de rostros generados por IA
- Los formatos de imagen admitidos son: jpg, jpeg, png y bmp

Si prefieres no descargar y organizar manualmente los datasets, puedes usar directamente la carpeta de datasets proporcionada en nuestro Google Drive, que ya sigue la estructura correcta.

## Modelos

### Nuestro Modelo

Implementamos un modelo CNN personalizado para la detección de imágenes generadas por IA. El modelo utiliza una arquitectura de 2 capas convolucionales seguidas de capas fully connected.

**Características principales:**
- Arquitectura ligera y eficiente
- Preprocesamiento de imágenes con detección y alineación de rostros
- Entrenado con un conjunto de datos de rostros reales y generados por IA

```bash
python model_ours.py
```

### Nuestro Modelo Fine-tuned

Al evaluar nuestro modelo original con un conjunto de datos diferente, observamos un rendimiento subóptimo. Por esto, realizamos fine-tuning con el nuevo conjunto de datos para mejorar la generalización.

**Mejoras:**
- Mayor precisión en diferentes conjuntos de datos
- Mejor capacidad de generalización a diferentes estilos de imágenes generadas por IA

```bash
python models_ours_finetunned.py
```

### CNN Detection

Implementamos el modelo CNN Detection de Wang et al., que utiliza una arquitectura ResNet50 para detectar imágenes generadas por IA.

**Características principales:**
- Arquitectura profunda basada en ResNet50
- Entrenado con un amplio conjunto de datos de imágenes sintéticas
- Alta precisión en la detección de artifacts de generación

```bash
python model_cnndetection.py
```

### FaceForensics

Implementamos el modelo FaceForensics, especializado en la detección de deepfakes y manipulaciones faciales.

**Características principales:**
- Basado en EfficientNet
- Especializado en detección de manipulaciones faciales
- Incluye detector de rostros integrado

```bash
python model_faceforensics.py
```

### FaceForensics Fine-tuned

Similar a nuestro enfoque con nuestro propio modelo, realizamos fine-tuning del modelo FaceForensics para mejorar su rendimiento con nuestro conjunto de datos.

**Mejoras:**
- Mayor precisión en la detección de rostros generados por IA modernos
- Mejor equilibrio entre detección de imágenes reales y falsas

```bash
python model_faceforensics_finetuned.py
```

## Comparación de modelos

Desarrollamos un script para comparar todos los modelos y visualizar sus métricas de rendimiento, incluyendo precisión, recall, F1-score y matrices de confusión.

```bash
python compare_models.py
```

Los resultados de la comparación se guardan en el directorio `comparison_results/`, que incluye gráficos y tablas mostrando el rendimiento relativo de cada modelo.

## Interfaz para análisis de imágenes

Implementamos una interfaz gráfica que permite cargar imágenes y analizarlas con todos los modelos disponibles, mostrando predicciones y visualizaciones.

**Características:**
- Carga de imágenes desde el sistema de archivos
- Detección y alineación automática de rostros
- Visualización de predicciones de todos los modelos
- Representación gráfica de resultados

```bash
python interface_models.py
```

## Referencias

- Nuestro proyecto utiliza modelos de los siguientes repositorios:
  - [DeepSafe](https://github.com/siddharthksah/DeepSafe): Recopilación de modelos de detección de deepfakes
  - [CNNDetection](https://github.com/peterwang512/CNNDetection): Detección de imágenes generadas por IA
  - [FaceForensics](https://github.com/ondyari/FaceForensics): Detección de manipulaciones faciales

---

Desarrollado por Héctor Paramio García e Iker Barrio Elorduy - 2025 