import cv2
import numpy as np
import os
from collections import deque

def region_growing(image, seed_point, threshold):
    """
    Realiza la segmentación de una imagen mediante el crecimiento de regiones.

    :param image: Imagen de entrada (se convertirá a escala de grises).
    :param seed_point: Tupla (y, x) con las coordenadas del píxel semilla.
    :param threshold: Umbral de diferencia de intensidad para incluir píxeles en la región.
    :return: Máscara binaria con la región segmentada.
    """
    # Obtener las dimensiones de la imagen
    height, width = image.shape[:2]
    
    # Asegurarse de que la imagen esté en escala de grises para la comparación
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Crear una máscara para la región segmentada, inicialmente todo en 0 (negro)
    segmented_mask = np.zeros((height, width), np.uint8)

    # Verificar que la semilla esté dentro de los límites de la imagen
    if not (0 <= seed_point[0] < height and 0 <= seed_point[1] < width):
        print("Error: El punto de semilla está fuera de los límites de la imagen.")
        return segmented_mask

    # Valor de intensidad de la semilla original
    seed_value = int(gray_image[seed_point[0], seed_point[1]])

    # Cola para los píxeles a procesar
    queue = deque([seed_point])

    # Marcar el punto de semilla en la máscara
    segmented_mask[seed_point[0], seed_point[1]] = 255

    # Definir los 8 vecinos posibles (conectividad-8)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        # Sacar un píxel de la cola
        current_y, current_x = queue.popleft()

        # Explorar los vecinos
        for dy, dx in neighbors:
            ny, nx = current_y + dy, current_x + dx

            # Comprobar si el vecino está dentro de los límites de la imagen
            if 0 <= ny < height and 0 <= nx < width:
                # Comprobar si el vecino ya ha sido visitado
                if segmented_mask[ny, nx] == 0:
                    # Obtener la intensidad del vecino
                    neighbor_value = int(gray_image[ny, nx])
                    
                    # Criterio de similitud: diferencia de intensidad con la semilla
                    if abs(neighbor_value - seed_value) < threshold:
                        # Añadir el vecino a la región
                        segmented_mask[ny, nx] = 255
                        queue.append((ny, nx))
    
    return segmented_mask

def main():
    """
    Función principal para cargar una imagen, ejecutar la segmentación y mostrar los resultados.
    """
    # --- Configuración ---
    image_filename = 'bocho.jpeg'
    
    # Construir la ruta a la imagen de forma robusta, relativa a la ubicación del script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script (src)
    project_root = os.path.dirname(script_dir)  # Raíz del proyecto
    image_path = os.path.join(project_root, 'images', image_filename)
    
    # Punto de semilla (y, x) - ¡Ajusta estas coordenadas para tu imagen!
    # Estas coordenadas se han elegido para 'coche08.jpg' para que caigan sobre el coche.
    seed_point = (800, 300)  
    
    # Umbral de similitud (0-255). Un valor más alto creará una región más grande.
    threshold = 40
    # ---------------------

    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
        return

    # Ejecutar el algoritmo de crecimiento de regiones
    print("Iniciando segmentación por crecimiento de regiones...")
    segmented_region = region_growing(image, seed_point, threshold)
    print("Segmentación completada.")

    # Crear una imagen de visualización donde la región segmentada se colorea de rojo
    # Convertir la imagen original a BGR si es necesario
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Superponer la máscara en color rojo sobre la imagen original
    overlay = image_bgr.copy()
    overlay[segmented_region == 255] = (0, 0, 255)  # BGR para rojo
    
    # Mezclar la imagen original con la superposición para un efecto de transparencia
    alpha = 0.6  # Transparencia
    result_image = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)

    # Marcar el punto de semilla en el resultado para referencia
    cv2.circle(result_image, (seed_point[1], seed_point[0]), 5, (0, 255, 0), -1, cv2.LINE_AA)

    # Mostrar las imágenes
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Region Segmentada', segmented_region)
    cv2.imshow('Resultado Superpuesto', result_image)

    print("Mostrando resultados. Presiona cualquier tecla en una de las ventanas para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
