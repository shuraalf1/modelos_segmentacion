
import cv2
import numpy as np
import os

# Predicado de homogeneidad: desviación estándar por debajo de un umbral
def is_homogeneous(region, std_dev_threshold):
    """
    Comprueba si una región es homogénea basándose en la desviación estándar de sus píxeles.
    """
    if region.size == 0:
        return True
    std_dev = np.std(region)
    return std_dev < std_dev_threshold

def split_and_merge(image, std_dev_threshold, min_region_size=4):
    """
    Realiza la segmentación de la imagen y devuelve tanto el resultado del split como del merge.
    """
    height, width = image.shape
    
    segmented_image = np.zeros_like(image, dtype=np.int32)
    current_label = 1
    regions_to_process = [(0, 0, height, width)]
    processed_regions = []

    # --- Fase de División (Split) ---
    while regions_to_process:
        y, x, h, w = regions_to_process.pop()
        region = image[y:y+h, x:x+w]

        if is_homogeneous(region, std_dev_threshold) or h <= min_region_size or w <= min_region_size:
            segmented_image[y:y+h, x:x+w] = current_label
            processed_regions.append((y, x, h, w, current_label))
            current_label += 1
        else:
            half_h, half_w = h // 2, w // 2
            if half_h > 0 and half_w > 0:
                regions_to_process.append((y, x, half_h, half_w))
                regions_to_process.append((y, x + half_w, half_h, w - half_w))
                regions_to_process.append((y + half_h, x, h - half_h, half_w))
                regions_to_process.append((y + half_h, x + half_w, h - half_h, w - half_w))
            else:
                segmented_image[y:y+h, x:x+w] = current_label
                processed_regions.append((y, x, h, w, current_label))
                current_label += 1

    # --- Visualización de la fase de Split ---
    # Normalizar las etiquetas para que se puedan visualizar como una imagen en escala de grises
    split_image_visual = np.zeros_like(image, dtype=np.uint8)
    if current_label > 1:
        # Se normalizan las etiquetas a un rango visible (0-255)
        split_image_visual = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- Fase de Unión (Merge) con Colores ---
    # Se crea una imagen en color para visualizar las regiones del merge
    colored_merged_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generar un color aleatorio para cada etiqueta de región
    labels = np.unique(segmented_image)
    colors = {label: np.random.randint(0, 255, 3).tolist() for label in labels if label != 0}

    # Colorear cada región en la imagen de merge
    for label, color in colors.items():
        colored_merged_image[segmented_image == label] = color

    return split_image_visual, colored_merged_image

def main():
    """
    Función principal para cargar la imagen, ejecutar el algoritmo y mostrar los resultados.
    """
    # --- Configuración ---
    image_filename = 'coche16.jpeg'
    
    # Construir la ruta a la imagen de forma robusta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, 'images', image_filename)
    
    print(f"Cargando imagen desde: {image_path}")
    
    # Cargar la imagen original en color
    image_color = cv2.imread(image_path)

    if image_color is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
        print("Asegúrate de que el archivo existe y la ruta es correcta.")
        return

    # Convertir a escala de grises para el algoritmo de segmentación
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # --- Parámetros del Algoritmo ---
    std_dev_threshold = 35
    min_region_size = 25

    print("Procesando la imagen con el método Split and Merge...")
    
    # Aplicar el algoritmo a la imagen en escala de grises
    split_result, merged_result = split_and_merge(image_gray, std_dev_threshold, min_region_size)

    print("Proceso completado.")

    # --- Superponer las regiones de colores sobre la imagen original ---
    # Se define el peso de la imagen original y de la máscara de color
    alpha = 0.6  # Peso de la imagen original
    beta = 0.4   # Peso de la máscara de segmentación
    gamma = 0    # Valor escalar añadido

    # Se crea la superposición
    overlay_result = cv2.addWeighted(image_color, alpha, merged_result, beta, gamma)

    # Mostrar resultados
    cv2.imshow('Imagen Original', image_color)
    cv2.imshow('Resultado del Split (Regiones)', split_result)
    cv2.imshow(f'Resultado Superpuesto (Umbral={std_dev_threshold})', overlay_result)

    print("Presiona cualquier tecla en una de las ventanas de imagen para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
