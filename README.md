# Modelos de Segmentación de Imágenes

Este proyecto contiene implementaciones de dos algoritmos clásicos de segmentación de imágenes: Crecimiento de Regiones (Region Growing) y División y Fusión (Split and Merge).

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

Los scripts se ejecutan directamente desde la línea de comandos. Las imágenes de entrada y los parámetros se configuran dentro de cada archivo de script.

### 1. Crecimiento de Regiones (`growing.py`)

Este algoritmo segmenta una región de una imagen a partir de un punto "semilla" inicial.

**Para ejecutar:**

```bash
python src/growing.py
```

**Configuración dentro de `src/growing.py`:**

-   `image_filename`: Cambia el nombre del archivo de la imagen que se encuentra en la carpeta `images/`.
-   `seed_point`: Modifica la tupla `(y, x)` para elegir un punto de inicio diferente para el crecimiento.
-   `threshold`: Ajusta el valor para controlar cuán similar debe ser un píxel vecino para ser incluido en la región. Un valor más alto permite que la región crezca más.

### 2. División y Fusión (`split_merge.py`)

Este algoritmo divide recursivamente la imagen en cuadrantes y luego los fusiona si son homogéneos, basándose en un umbral.

**Para ejecutar:**

```bash
python src/split_merge.py
```

**Configuración dentro de `src/split_merge.py`:**

-   `image_filename`: Cambia el nombre del archivo de la imagen que se encuentra en la carpeta `images/`.
-   `threshold`: Ajusta el umbral de homogeneidad. Un valor más bajo resultará en más divisiones y una segmentación más detallada.

El resultado de la segmentación se guardará como `segmented_image_with_overlay.png` en el directorio raíz del proyecto.
