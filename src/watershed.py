import cv2
import numpy as np
import os

def watershed_segmentation(image):
    """
    Realiza la segmentación de una imagen utilizando el algoritmo Watershed de OpenCV.

    :param image: Imagen de entrada en formato BGR.
    :return: Una tupla con la imagen de los marcadores y la imagen con los resultados de la segmentación.
    """
    print("Iniciando segmentación con Watershed...")

    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Umbralización para separar objetos del fondo
    # Usamos el método de Otsu para encontrar un umbral óptimo automáticamente.
    # Los objetos de interés (más claros) quedarán en blanco.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Eliminación de ruido (apertura morfológica)
    # Esto elimina pequeños puntos blancos en el fondo.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Identificar el área segura del fondo (sure background)
    # Dilatamos la imagen para estar seguros de qué es fondo.
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Identificar el área segura del primer plano (sure foreground)
    # La transformada de distancia nos da la distancia de cada píxel al fondo.
    # Los picos de esta transformada son los centros de los objetos.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # --- PARÁMETRO CLAVE ---
    # Reduce este valor (ej. de 0.5 a 0.2) para detectar más regiones (más pequeñas).
    # Un valor más bajo es más sensible para encontrar centros de objetos.
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 6. Identificar la región desconocida (bordes)
    # Es la diferencia entre el fondo seguro y el primer plano seguro.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Crear los marcadores para el algoritmo Watershed
    # Etiquetamos las regiones de primer plano seguro con números positivos.
    _, markers = cv2.connectedComponents(sure_fg)
    # Sumamos 1 para que el fondo (etiqueta 0) se convierta en 1.
    markers = markers + 1
    # Marcamos la región desconocida con 0. El algoritmo debe decidir a qué región pertenecen.
    markers[unknown == 255] = 0

    # 8. Aplicar el algoritmo Watershed
    # El algoritmo llenará las cuencas a partir de los marcadores.
    markers = cv2.watershed(image, markers)

    # 9. Visualizar los resultados
    # Los bordes encontrados por Watershed se marcan con -1.
    # Creamos una imagen de resultado donde los bordes son rojos.
    result_image = image.copy()
    result_image[markers == -1] = [0, 0, 255]  # BGR para rojo

    # Convertir la imagen de marcadores a un formato visible
    markers_display = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    markers_display = cv2.applyColorMap(markers_display, cv2.COLORMAP_JET)
    markers_display[markers == -1] = [0, 0, 255]

    print("Segmentación con Watershed completada.")
    return markers_display, result_image

def main():
    """
    Función principal para cargar una imagen, ejecutar la segmentación y mostrar los resultados.
    """
    
    # --- Configuración ---
    name ='Cars11'
    typeImg = '.png'
    image_filename = name + typeImg
    
    # Construir la ruta a la imagen de forma robusta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, 'images', image_filename)
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
        return

    # Ejecutar el algoritmo Watershed
    segmented_image, markers_image = watershed_segmentation(image)

    # Mostrar las imágenes
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Marcadores para Watershed', markers_image)
    cv2.imshow('Resultado de Watershed', segmented_image)
     # Guardar los resultados
    output_dir = os.path.join(project_root, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, f"{name}_watershed_marcadores.png"), markers_image)
    cv2.imwrite(os.path.join(output_dir, f'{name}_watershed_region_segmentada.png'), segmented_image)

    print("Mostrando resultados. Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   
if __name__ == "__main__":
    main()