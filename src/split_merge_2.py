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

def merge_regions(segmented_image, merge_threshold):
    """
    Fusiona regiones adyacentes si su intensidad media es similar.

    :param segmented_image: La imagen en escala de grises después de la fase de división.
    :param merge_threshold: La diferencia máxima de intensidad media para fusionar regiones.
    :return: Una nueva imagen en escala de grises con las regiones fusionadas.
    """
    print("Iniciando la fase de Merge...")
    
    # 1. Etiquetar componentes conectados para obtener las regiones iniciales
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8, ltype=cv2.CV_32S)

    # Calcular el valor promedio de gris para cada región inicial
    # (El valor ya es uniforme, así que podemos tomar el de cualquier píxel)
    initial_means = np.zeros(num_labels)
    for label_id in range(1, num_labels):
        # Tomamos el valor de gris del centroide de la región
        centroid_x, centroid_y = map(int, centroids[label_id])
        initial_means[label_id] = segmented_image[centroid_y, centroid_x]

    # 2. Construir el grafo de adyacencia de regiones (RAG)
    adjacencies = [set() for _ in range(num_labels)]
    height, width = labels.shape
    for y in range(height - 1):
        for x in range(width - 1):
            # Comprobar vecino derecho
            l1 = labels[y, x]
            l2 = labels[y, x + 1]
            if l1 != l2:
                adjacencies[l1].add(l2)
                adjacencies[l2].add(l1)
            # Comprobar vecino inferior
            l2 = labels[y + 1, x]
            if l1 != l2:
                adjacencies[l1].add(l2)
                adjacencies[l2].add(l1)

    # 3. Fusión iterativa
    # `merged_into` mapea una etiqueta a la etiqueta en la que se ha fusionado
    merged_into = list(range(num_labels))
    
    for label1 in range(1, num_labels):
        for label2 in list(adjacencies[label1]):
            # Encontrar la raíz de la fusión para cada etiqueta
            root1 = label1
            while merged_into[root1] != root1:
                root1 = merged_into[root1]
            
            root2 = label2
            while merged_into[root2] != root2:
                root2 = merged_into[root2]

            if root1 != root2:
                mean1 = initial_means[root1]
                mean2 = initial_means[root2]
                
                if abs(mean1 - mean2) <= merge_threshold:
                    # Fusionar la región más pequeña en la más grande
                    if stats[root1, cv2.CC_STAT_AREA] < stats[root2, cv2.CC_STAT_AREA]:
                        merged_into[root1] = root2
                    else:
                        merged_into[root2] = root1

    # 4. Renderizar la imagen final
    final_image = np.zeros_like(segmented_image)
    for label_id in range(1, num_labels):
        root = label_id
        while merged_into[root] != root:
            root = merged_into[root]
        final_image[labels == label_id] = initial_means[root]
    
    print("Fase de Merge completada.")
    return final_image

def find_region_at_point(segmented_image, seed_point):
    """
    Encuentra la región específica que contiene el seed_point de una imagen pre-segmentada.

    :param segmented_image: La imagen en escala de grises después de la división y fusión.
    :param seed_point: Una tupla (y, x) para el píxel semilla.
    :return: Una máscara binaria de la región específica.
    """
    y, x = seed_point
    if not (0 <= y < segmented_image.shape[0] and 0 <= x < segmented_image.shape[1]):
        print("Error: El punto de semilla está fuera de los límites de la imagen.")
        return np.zeros_like(segmented_image, dtype=np.uint8)

    region_value = segmented_image[y, x]
    mask = np.where(segmented_image == region_value, 255, 0).astype(np.uint8)
    return mask

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
        print(f"Error: No se pudo cargar la imagen desde {image_path}")
        return

    # Umbral para la fase de SPLIT (división)
    # Un valor más alto crea bloques iniciales más grandes.
    split_threshold = 10

    # Umbral para la fase de MERGE (fusión)
    # Un valor más alto fusionará regiones con colores más diferentes.
    merge_threshold = 20

    # Punto de semilla (y, x) - ¡Ajusta estas coordenadas para tu imagen!
    seed_point = (210, 200)

    # 1. Fase de Split
    print("Iniciando segmentación con Split...")
    split_result = split_and_merge(image, split_threshold)
    
    # 2. Fase de Merge
    merged_result = merge_regions(split_result, merge_threshold)

    # Find the specific region at the seed point
    region_mask = find_region_at_point(merged_result, seed_point)

    # Crea una copia de la imagen original para dibujar sobre ella
    output = image.copy()

    # Create a red overlay for the selected region
    output[region_mask == 255] = (0, 0, 255) # BGR for red

    # Mezcla la superposición con la imagen original
    alpha = 0.6  # Factor de transparencia
    cv2.addWeighted(output, alpha, image.copy(), 1 - alpha, 0, output)

    # Muestra las imágenes original y segmentada
    cv2.imshow('Original Image', image)
    cv2.imshow('Resultado solo Split', split_result)
    cv2.imshow('Resultado Final (Split & Merge)', merged_result)
    cv2.imshow('Region Especifica con Transparencia', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
