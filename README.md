# Modelos de Segmentación de Imágenes

Este proyecto contiene implementaciones de algoritmos de segmentación de imágenes, incluyendo Crecimiento de Regiones con detección de semillas, Watershed, y un script para evaluar los resultados contra un ground truth.

## Requisitos

- Python 3.x
- Las librerías especificadas en `requirements.txt`.

## Instalación

1.  **Clonar el repositorio (si aplica):**
    ```bash
    git clone <URL-del-repositorio>
    cd modelos_segmentacion
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    ```
    Para activarlo:
    - En Windows: `.\venv\Scripts\activate`
    - En macOS/Linux: `source venv/bin/activate`

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución de los Modelos

Los scripts se ejecutan directamente desde la línea de comandos. Las imágenes de entrada y los parámetros se configuran dentro de cada archivo de script en la carpeta `src/`.

### 1. Crecimiento de Regiones con Detección de Semilla (`growing-seed-filtro.py`)

Este algoritmo segmenta una región de una imagen a partir de un punto "semilla". El script primero intenta detectar una matrícula de coche (un rectángulo) y usa su centro como semilla. Si no detecta ninguna, usa una semilla por defecto.

**Para ejecutar:**

```bash
python src/growing-seed-filtro.py
```

**Configuración dentro de `src/growing-seed-filtro.py`:**

-   `image_filename`: Cambia el nombre del archivo de la imagen que se encuentra en la carpeta `images/`.
-   `seed_point`: Coordenadas `(y, x)` de la semilla por defecto si no se detecta ningún rectángulo.
-   `threshold`: Ajusta el valor para controlar cuán similar debe ser un píxel vecino para ser incluido en la región. Un valor más alto permite que la región crezca más.

### 2. Segmentación con Watershed (`watershed.py`)

Este script utiliza el algoritmo Watershed de OpenCV para segmentar una imagen, separando los objetos del fondo.

**Para ejecutar:**

```bash
python src/watershed.py
```

**Configuración dentro de `src/watershed.py`:**

-   `image_filename`: Cambia el nombre del archivo de la imagen en la carpeta `images/`.
-   El umbral para la transformada de distancia (`0.2 * dist_transform.max()`) es un parámetro clave. Reducir el multiplicador (e.g., a `0.1`) puede ayudar a detectar más regiones o más pequeñas.

### 3. Evaluación de Segmentación (`segmentadores-vs-groundthru.py`)

Este script compara una máscara de segmentación generada (la predicción) con una máscara de ground truth para evaluar su rendimiento. Calcula métricas como IoU, Dice, Precisión y Recall.

**Para ejecutar:**

```bash
python src/segmentadores-vs-groundthru.py
```

**Configuración dentro de `src/segmentadores-vs-groundthru.py`:**

-   `ruta_predicha`: Ruta a la imagen de la máscara de segmentación que quieres evaluar.
-   `ruta_ground_truth`: Ruta a la imagen de la máscara de ground truth.

El script muestra una comparación visual y guarda un resumen de las métricas en `resultados_evaluacion.txt`.

## Salidas

Los scripts guardan las imágenes resultantes (imágenes con rectángulos detectados, regiones segmentadas, etc.) en la carpeta `output/`.