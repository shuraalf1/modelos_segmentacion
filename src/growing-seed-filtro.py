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

def aplicar_filtros_y_detectar_rectangulos(image):
    """
    Aplica filtros a la imagen y detecta rectángulos, devolviendo las coordenadas del centro
    y una imagen con los rectángulos dibujados.
    
    :param image: Imagen de entrada
    :return: Tuple (centros_rectangulos, imagen_con_rectangulos)
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) > 2:
        img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gris = image.copy()
    
    # Crear una copia de la imagen original para dibujar los rectángulos
    imagen_con_rectangulos = image.copy()
    
    centros_rectangulos = []
    rectangulos_info = []  # Guardar información de cada rectángulo
    
    # Paso 1. Suavizar la imagen (filtro gaussiano con kernel de 11x11)
    blur = cv2.GaussianBlur(img_gris, (11, 11), 0)  
    
    # Paso 2. Mejora de contraste (ajuste local de contraste adaptativo)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  
    equ01 = clahe.apply(blur)  
    
    # Paso 3. Binarizar la imagen (Otsu)
    otsu_threshold, otsu01 = cv2.threshold(equ01, 0, 255, cv2.THRESH_OTSU)  
    
    # Paso 4. Obtener bordes (Canny)
    edges = cv2.Canny(otsu01, 150, 350)  
    
    # Paso 5. Encontrar contornos
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    
    # Filtrar contornos por longitud
    long_contours = [cnt for cnt in contours if (cv2.arcLength(cnt, True) > 200 
                       and cv2.arcLength(cnt, True) < 700)]  
    
    # Paso 6. Detectar rectángulos y calcular centros
    alto, ancho = img_gris.shape
    for i, lonc in enumerate(long_contours):
        # Extraer coordenadas x e y del contorno
        r = [x for [[x,y]] in lonc]
        s = [y for [[x,y]] in lonc]
        prop = (max(s)-min(s))-(max(r)-min(r))
        
        # Filtro para detectar rectángulos horizontales en la parte inferior
        if ((max(s) > round(alto/2)) and prop < 0):
            # Coordenadas del rectángulo
            x_min, y_min = min(r), min(s)
            x_max, y_max = max(r), max(s)
            
            # Calcular centro del rectángulo
            centro_x = (x_min + x_max) // 2
            centro_y = (y_min + y_max) // 2
            #Punto 75% para calculo de semilla

            centro_x = (x_min + centro_x) // 2
            centro_y = (y_min + centro_y) // 2
            
            # Guardar coordenadas del centro e información del rectángulo
            centros_rectangulos.append((centro_x, centro_y))
            rectangulos_info.append({
                'id': i + 1,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'centro_x': centro_x,
                'centro_y': centro_y
            })
            
            # Dibujar rectángulo en la imagen
            cv2.rectangle(imagen_con_rectangulos, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Dibujar punto en el centro
            cv2.circle(imagen_con_rectangulos, (centro_x, centro_y), 5, (0, 0, 255), -1)
            
            # Añadir texto con las coordenadas del centro
            texto_coordenadas = f"({centro_x}, {centro_y})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            escala_fuente = 0.6
            grosor_fuente = 2
            
            # Dibujar fondo para el texto
            (ancho_texto, alto_texto), baseline = cv2.getTextSize(texto_coordenadas, font, escala_fuente, grosor_fuente)
            cv2.rectangle(imagen_con_rectangulos, 
                         (centro_x - 5, centro_y - alto_texto - 10), 
                         (centro_x + ancho_texto + 5, centro_y), 
                         (255, 255, 255), -1)
            
            # Dibujar texto
            cv2.putText(imagen_con_rectangulos, texto_coordenadas, 
                       (centro_x, centro_y - 5), 
                       font, escala_fuente, (255, 0, 0), grosor_fuente)
            
            print(f"Rectángulo {i+1} detectado - Centro: ({centro_x}, {centro_y})")
    
    print(f"Umbral Otsu: {otsu_threshold}")
    print(f"Contornos encontrados: {len(contours)}")
    print(f"Contornos filtrados: {len(long_contours)}")
    print(f"Rectángulos detectados: {len(centros_rectangulos)}")
    
    return centros_rectangulos, imagen_con_rectangulos, rectangulos_info

def mostrar_imagenes_ordenadas(images_dict):
    """
    Muestra las imágenes en posiciones ordenadas en la pantalla.
    
    :param images_dict: Diccionario con {nombre_ventana: imagen}
    """
    # Configuración de la cuadrícula
    filas = 2
    columnas = 4
    ancho_ventana = 400
    alto_ventana = 300
    margen_x = 20
    margen_y = 40
    
    # Obtener lista de nombres de ventanas
    nombres_ventanas = list(images_dict.keys())
    
    for i, nombre_ventana in enumerate(nombres_ventanas):
        # Calcular posición en la cuadrícula
        fila = i // columnas
        columna = i % columnas
        
        # Calcular coordenadas x, y
        x = columna * (ancho_ventana + margen_x) + margen_x
        y = fila * (alto_ventana + margen_y) + margen_y
        
        # Redimensionar imagen si es muy grande
        img = images_dict[nombre_ventana]
        h, w = img.shape[:2]
        
        # Calcular factor de escala para que quepa en la ventana
        escala = min(ancho_ventana / w, alto_ventana / h, 1.0)
        nuevo_w = int(w * escala)
        nuevo_h = int(h * escala)
        
        if escala < 1.0:
            img_redimensionada = cv2.resize(img, (nuevo_w, nuevo_h))
        else:
            img_redimensionada = img
        
        # Mostrar ventana en posición específica
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_ventana, nuevo_w, nuevo_h)
        cv2.moveWindow(nombre_ventana, x, y)
        cv2.imshow(nombre_ventana, img_redimensionada)

def main():
    """
    Función principal para cargar una imagen, ejecutar la segmentación y mostrar los resultados.
    """
    # --- Configuración ---
    name ='Cars39'
    typeImg = '.png'
    image_filename = name + typeImg
    
    # Construir la ruta a la imagen de forma robusta, relativa a la ubicación del script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script (src)
    project_root = os.path.dirname(script_dir)  # Raíz del proyecto
    image_path = os.path.join(project_root, 'images', image_filename)
    
    # Punto de semilla por defecto (y, x)
    seed_point = (300, 254)
    
    # Umbral de similitud (0-255). Un valor más alto creará una región más grande.
    threshold = 40
    # ---------------------

    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
        return

    # Aplicar filtros y detectar rectángulos
    print("Aplicando filtros y detectando rectángulos...")
    centros_rectangulos, imagen_con_rectangulos, rectangulos_info = aplicar_filtros_y_detectar_rectangulos(image)
    
    # Si se detectaron rectángulos, usar el primer centro como semilla
    if centros_rectangulos:
        centro_x, centro_y = centros_rectangulos[0]
        seed_point = (centro_y, centro_x)  # Convertir a formato (y, x)
        print(f"Usando centro del rectángulo como semilla: {seed_point}")
    else:
        print("No se detectaron rectángulos. Usando semilla por defecto.")

    # Ejecutar el algoritmo de crecimiento de regiones
    print("Iniciando segmentación por crecimiento de regiones...")
    segmented_region = region_growing(image, seed_point, threshold)
    print("Segmentación completada.")

    # Crear una imagen de visualización donde la región segmentada se colorea de rojo
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
    
    # Mostrar las coordenadas del centro en la imagen resultante
    if centros_rectangulos:
        for i, (centro_x, centro_y) in enumerate(centros_rectangulos):
            # Texto con las coordenadas
            texto_coordenadas = f"Centro {i+1}: ({centro_x}, {centro_y})"
            
            # Configuración del texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            escala_fuente = 0.5
            grosor_fuente = 1
            color_texto = (255, 255, 0)  # Cian
            
            # Dibujar punto en el centro (en la imagen resultante también)
            cv2.circle(result_image, (centro_x, centro_y), 5, (0, 255, 255), -1)
            
            # Dibujar texto con las coordenadas
            cv2.putText(result_image, texto_coordenadas, 
                      (centro_x + 10, centro_y + 10 * (i + 1)), 
                      font, escala_fuente, color_texto, grosor_fuente)

    # Preparar imágenes intermedias de filtros
    if len(image.shape) > 2:
        img_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gris = image.copy()
    
    blur = cv2.GaussianBlur(img_gris, (11, 11), 0)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  
    equ01 = clahe.apply(blur)
    _, otsu01 = cv2.threshold(equ01, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(otsu01, 150, 350)
    
    # Crear diccionario con todas las imágenes a mostrar
    imagenes_a_mostrar = {
        '1. Imagen Original': image,
        '2. Rectangulos Detectados': imagen_con_rectangulos,
        '3. Region Segmentada': segmented_region,
      #  '4. Resultado Final': result_image,
      #  '5. Filtro Gaussiano': blur,
      #  '6. Mejora Contraste': equ01,
      #  '7. Binarizacion Otsu': otsu01,
      #  '8. Bordes Canny': edges
    }
    
    # Mostrar todas las imágenes ordenadas
    mostrar_imagenes_ordenadas(imagenes_a_mostrar)
    # Guardar los resultados
    output_dir = os.path.join(project_root, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Guardar resultados
    cv2.imwrite('rectangulos_detectados.png', imagen_con_rectangulos)
    cv2.imwrite('region_segmentada.png', segmented_region)
    cv2.imwrite('resultado_final.png', result_image)
    cv2.imwrite(os.path.join(output_dir, f"{name}_growing_rectangulos_detectados.png"), imagen_con_rectangulos)
    cv2.imwrite(os.path.join(output_dir, f'{name}_growing_region_segmentada.png'), segmented_region)
    cv2.imwrite(os.path.join(output_dir, f'{name}_growing_resultado_final.png'), result_image)

    print("\nResumen de ventanas (ordenadas en cuadrícula 4x2):")
    for i, nombre in enumerate(imagenes_a_mostrar.keys(), 1):
        print(f"{nombre}")
    
    print("\nPresiona cualquier tecla en una de las ventanas para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()