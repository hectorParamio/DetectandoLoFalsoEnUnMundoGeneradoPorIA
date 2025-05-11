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
git clone https://github.com/TU_USUARIO/detector-imagenes-ia.git
cd detector-imagenes-ia

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelos pre-entrenados (si es necesario)
python download_models.py
```

## Estructura del proyecto

- `model_ours.py`: Nuestro modelo CNN para detección de imágenes generadas por IA
- `models_ours_finetunned.py`: Versión fine-tuned de nuestro modelo
- `model_cnndetection.py`: Implementación del modelo CNN Detection
- `model_faceforensics.py`: Implementación del modelo FaceForensics
- `model_faceforensics_finetuned.py`: Versión fine-tuned del modelo FaceForensics
- `compare_models.py`: Script para comparar el rendimiento de todos los modelos
- `interface_models.py`: Interfaz gráfica para analizar imágenes con todos los modelos

## Modelos

### Nuestro Modelo

Implementamos un modelo CNN personalizado para la detección de imágenes generadas por IA. El modelo utiliza una arquitectura de 2 capas convolucionales seguidas de capas fully connected.

**Características principales:**
- Arquitectura ligera y eficiente
- Preprocesamiento de imágenes con detección y alineación de rostros
- Entrenado con un conjunto de datos de rostros reales y generados por IA

```bash
# Ejecutar entrenamiento
python model_ours.py
```

### Nuestro Modelo Fine-tuned

Al evaluar nuestro modelo original con un conjunto de datos diferente, observamos un rendimiento subóptimo. Por esto, realizamos fine-tuning con el nuevo conjunto de datos para mejorar la generalización.

**Mejoras:**
- Mayor precisión en diferentes conjuntos de datos
- Mejor capacidad de generalización a diferentes estilos de imágenes generadas por IA

```bash
# Ejecutar fine-tuning
python models_ours_finetunned.py
```

### CNN Detection

Implementamos el modelo CNN Detection de Wang et al., que utiliza una arquitectura ResNet50 para detectar imágenes generadas por IA.

**Características principales:**
- Arquitectura profunda basada en ResNet50
- Entrenado con un amplio conjunto de datos de imágenes sintéticas
- Alta precisión en la detección de artifacts de generación

```bash
# Ejecutar evaluación
python model_cnndetection.py
```

### FaceForensics

Implementamos el modelo FaceForensics, especializado en la detección de deepfakes y manipulaciones faciales.

**Características principales:**
- Basado en EfficientNet
- Especializado en detección de manipulaciones faciales
- Incluye detector de rostros integrado

```bash
# Ejecutar evaluación
python model_faceforensics.py
```

### FaceForensics Fine-tuned

Similar a nuestro enfoque con nuestro propio modelo, realizamos fine-tuning del modelo FaceForensics para mejorar su rendimiento con nuestro conjunto de datos.

**Mejoras:**
- Mayor precisión en la detección de rostros generados por IA modernos
- Mejor equilibrio entre detección de imágenes reales y falsas

```bash
# Ejecutar fine-tuning
python model_faceforensics_finetuned.py
```

## Comparación de modelos

Desarrollamos un script para comparar todos los modelos y visualizar sus métricas de rendimiento, incluyendo precisión, recall, F1-score y matrices de confusión.

```bash
# Ejecutar comparación
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
# Lanzar la interfaz
python interface_models.py
```

## Referencias

- Nuestro proyecto utiliza modelos de los siguientes repositorios:
  - [DeepSafe](https://github.com/siddharthksah/DeepSafe): Recopilación de modelos de detección de deepfakes
  - [CNNDetection](https://github.com/peterwang512/CNNDetection): Detección de imágenes generadas por IA
  - [FaceForensics](https://github.com/ondyari/FaceForensics): Detección de manipulaciones faciales

---

Desarrollado por [Tu Nombre] y [Nombre de tu Compañero] - 2024 