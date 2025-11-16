import cv2
import numpy as np
import os
def is_homogeneous(region, threshold):
    """Comprueba si la región es homogénea según el umbral de intensidad."""
    min_val, max_val = np.min(region), np.max(region)
    return (max_val - min_val) <= threshold

def split_and_merge(image, threshold):
    """Segmenta la imagen dividiendo y fusionando regiones de forma recursiva."""

    def recursive_split(region):
        rows, cols = region.shape
        if rows <= 1 or cols <= 1:
            return np.zeros_like(region, dtype=np.uint8)

        if is_homogeneous(region, threshold):
            # Si es homogénea, devuelve una región rellena con la intensidad media
            return np.full_like(region, int(np.mean(region)), dtype=np.uint8)

        # Split the region into four quadrants
        mid_row, mid_col = rows // 2, cols // 2

        top_left = region[:mid_row, :mid_col]
        top_right = region[:mid_row, mid_col:]
        bottom_left = region[mid_row:, :mid_col]
        bottom_right = region[mid_row:, mid_col:]

        # Divide recursivamente cada cuadrante
        split_top_left = recursive_split(top_left)
        split_top_right = recursive_split(top_right)
        split_bottom_left = recursive_split(bottom_left)
        split_bottom_right = recursive_split(bottom_right)

        # Combina los resultados
        top_half = np.hstack([split_top_left, split_top_right])
        bottom_half = np.hstack([split_bottom_left, split_bottom_right])
        return np.vstack([top_half, bottom_half])

    # Asegura que la imagen esté en escala de grises
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Aplica el algoritmo de división y fusión
    segmented_image = recursive_split(gray_image)

    return segmented_image

def main():
    # --- Configuración ---
    image_filename = 'Cars11.png'
    
    # Construir la ruta a la imagen de forma robusta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, 'images', image_filename)
    
    print(f"Cargando imagen desde: {image_path}")
    
    # Cargar la imagen original en color
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {file_path}")
        return

    if image is None:
        print("Error: Image could not be loaded from local path or URL.")
        return

    # Establece el umbral de homogeneidad
    threshold = 10  # Ajusta este valor según sea necesario

    # Segmenta la imagen
    segmented_result = split_and_merge(image, threshold)

    # Crea una representación visual de la segmentación
    # Encuentra los contornos de las regiones segmentadas
    contours, _ = cv2.findContours(segmented_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una copia de la imagen original para dibujar sobre ella
    overlay = image.copy()
    output = image.copy()

    # Genera colores aleatorios para cada región
    for i, contour in enumerate(contours):
        color = np.random.randint(0, 255, size=3).tolist()
        # Dibuja el contorno relleno en la superposición
        cv2.drawContours(overlay, [contour], -1, color, -1)

    # Mezcla la superposición con la imagen original
    alpha = 0.6  # Factor de transparencia
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)


    # Muestra las imágenes original y segmentada
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Regions with Transparency', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guarda el resultado
    cv2.imwrite('segmented_image_with_overlay.png', output)

if __name__ == "__main__":
    main()
