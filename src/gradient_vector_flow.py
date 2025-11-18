import cv2
import numpy as np
import os

def mostrar_imagenes_ordenadas(images_dict):
    """
    Muestra las imágenes en posiciones ordenadas en la pantalla.
    
    :param images_dict: Diccionario con {nombre_ventana: imagen}
    """
    # Configuración de la cuadrícula
    filas = 2
    columnas = 3
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

def calcular_gvf(campo_gradiente, mu=0.2, iteraciones=80):
    """
    Calcula el Gradient Vector Flow (GVF) a partir de un campo de gradiente.
    
    :param campo_gradiente: Campo de gradiente inicial (u, v)
    :param mu: Parámetro de regularización
    :param iteraciones: Número de iteraciones para la difusión
    :return: Campo vectorial GVF (u_gvf, v_gvf)
    """
    print("Calculando Gradient Vector Flow...")
    
    # Separar componentes del gradiente
    u = campo_gradiente[:, :, 0].astype(np.float64)
    v = campo_gradiente[:, :, 1].astype(np.float64)
    
    # Calcular la magnitud del gradiente al cuadrado
    grad_mag_sq = u**2 + v**2
    
    # Inicializar GVF con el gradiente original
    u_gvf = u.copy()
    v_gvf = v.copy()
    
    # Iterar para difundir el campo vectorial
    for i in range(iteraciones):
        # Laplaciano (difusión)
        laplaciano_u = cv2.Laplacian(u_gvf, cv2.CV_64F)
        laplaciano_v = cv2.Laplacian(v_gvf, cv2.CV_64F)
        
        # Actualizar GVF según la ecuación
        u_gvf = u_gvf + mu * laplaciano_u - grad_mag_sq * (u_gvf - u)
        v_gvf = v_gvf + mu * laplaciano_v - grad_mag_sq * (v_gvf - v)
        
        if i % 20 == 0:
            print(f"  Iteración {i}/{iteraciones}")
    
    print("Cálculo de GVF completado.")
    return np.stack([u_gvf, v_gvf], axis=2)

def crecimiento_regiones_gvf(imagen, semillas, parametros):
    """
    Realiza crecimiento de regiones usando Gradient Vector Flow.
    
    :param imagen: Imagen de entrada en escala de grises
    :param semillas: Lista de puntos (y, x) para semillas
    :param parametros: Diccionario con parámetros del algoritmo
    :return: Máscara de segmentación
    """
    print("Iniciando crecimiento de regiones con GVF...")
    
    # Extraer parámetros
    umbral_gvf = parametros.get('umbral_gvf', 0.1)
    umbral_intensidad = parametros.get('umbral_intensidad', 20)
    tamano_maximo = parametros.get('tamano_maximo', 10000)
    
    # 1. Calcular gradiente de la imagen
    grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    
    # Crear campo de gradiente inicial
    campo_gradiente = np.stack([grad_x, grad_y], axis=2)
    
    # 2. Calcular GVF
    campo_gvf = calcular_gvf(campo_gradiente)
    
    # 3. Inicializar máscara de segmentación
    h, w = imagen.shape
    mascara = np.zeros((h, w), dtype=np.uint8)
    visitado = np.zeros((h, w), dtype=bool)
    
    # 4. Procesar cada semilla
    for i, (y_semilla, x_semilla) in enumerate(semillas):
        print(f"Procesando semilla {i+1}/{len(semillas)} en ({x_semilla}, {y_semilla})")
        
        # Verificar que la semilla esté dentro de la imagen
        if not (0 <= y_semilla < h and 0 <= x_semilla < w):
            print(f"  Semilla fuera de los límites, saltando...")
            continue
        
        # Inicializar región para esta semilla
        region_actual = np.zeros((h, w), dtype=bool)
        cola = [(y_semilla, x_semilla)]
        intensidad_referencia = imagen[y_semilla, x_semilla]
        
        region_actual[y_semilla, x_semilla] = True
        visitado[y_semilla, x_semilla] = True
        
        while cola and np.sum(region_actual) < tamano_maximo:
            y, x = cola.pop(0)
            
            # Vecinos 4-conectados
            vecinos = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
            
            for ny, nx in vecinos:
                # Verificar límites
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                
                # Verificar si ya fue visitado en esta región
                if region_actual[ny, nx] or visitado[ny, nx]:
                    continue
                
                # Criterio 1: Similitud de intensidad
                diff_intensidad = abs(int(imagen[ny, nx]) - int(intensidad_referencia))
                if diff_intensidad > umbral_intensidad:
                    continue
                
                # Criterio 2: Flujo del campo GVF
                # Calcular producto punto entre dirección de crecimiento y GVF
                direccion = np.array([ny - y, nx - x])
                direccion_norm = direccion / (np.linalg.norm(direccion) + 1e-8)
                
                gvf_punto = campo_gvf[ny, nx, 0] * direccion_norm[1] + \
                           campo_gvf[ny, nx, 1] * direccion_norm[0]
                
                # El crecimiento debe seguir la dirección del campo GVF
                if gvf_punto < umbral_gvf:
                    continue
                
                # Agregar píxel a la región
                region_actual[ny, nx] = True
                visitado[ny, nx] = True
                cola.append((ny, nx))
        
        # Agregar región a la máscara final
        mascara[region_actual] = 255
        
        print(f"  Región {i+1} creció a {np.sum(region_actual)} píxeles")
    
    print("Crecimiento de regiones con GVF completado.")
    return mascara

def gvf_region_growing_segmentation(imagen, semillas_manual=None, parametros=None):
    """
    Segmentación completa usando crecimiento de regiones con GVF.
    
    :param imagen: Imagen de entrada en formato BGR
    :param semillas_manual: Lista de semillas manuales [(y1, x1), (y2, x2), ...]
    :param parametros: Parámetros para el algoritmo
    :return: Tupla con imágenes intermedias y resultado final
    """
    # Parámetros por defecto
    if parametros is None:
        parametros = {
            'umbral_gvf': 0.1,
            'umbral_intensidad': 20,
            'tamano_maximo': 10000
        }
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Si no se proporcionan semillas, detectar automáticamente
    if semillas_manual is None:
        semillas = detectar_semillas_automaticas(gris)
    else:
        semillas = semillas_manual
    
    # Aplicar crecimiento de regiones con GVF
    mascara_segmentacion = crecimiento_regiones_gvf(gris, semillas, parametros)
    
    # Aplicar la máscara a la imagen original
    resultado = imagen.copy()
    resultado[mascara_segmentacion == 0] = 0
    
    # Crear visualizaciones intermedias
    # 1. Gradiente original
    grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    magnitud_gradiente = np.sqrt(grad_x**2 + grad_y**2)
    vis_gradiente = cv2.normalize(magnitud_gradiente, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_gradiente = cv2.applyColorMap(vis_gradiente, cv2.COLORMAP_JET)
    
    # 2. Campo GVF (visualización)
    campo_gradiente = np.stack([grad_x, grad_y], axis=2)
    campo_gvf = calcular_gvf(campo_gradiente)
    magnitud_gvf = np.sqrt(campo_gvf[:,:,0]**2 + campo_gvf[:,:,1]**2)
    vis_gvf = cv2.normalize(magnitud_gvf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_gvf = cv2.applyColorMap(vis_gvf, cv2.COLORMAP_JET)
    
    # 3. Imagen con semillas marcadas
    vis_semillas = imagen.copy()
    for y, x in semillas:
        cv2.circle(vis_semillas, (x, y), 5, (0, 255, 0), -1)  # Puntos verdes
    
    return vis_gradiente, vis_gvf, vis_semillas, mascara_segmentacion, resultado

def detectar_semillas_automaticas(imagen_gris, num_semillas=5):
    """
    Detecta semillas automáticamente usando detección de esquinas.
    
    :param imagen_gris: Imagen en escala de grises
    :param num_semillas: Número de semillas a detectar
    :return: Lista de coordenadas (y, x) de semillas
    """
    # Usar Harris corner detection
    esquinas = cv2.cornerHarris(imagen_gris, blockSize=2, ksize=3, k=0.04)
    
    # Umbralizar para obtener las esquinas más fuertes
    umbral = 0.01 * esquinas.max()
    coordenadas = np.argwhere(esquinas > umbral)
    
    # Si hay muchas esquinas, seleccionar las más fuertes
    if len(coordenadas) > num_semillas:
        # Ordenar por fuerza de la esquina
        fuerzas = [esquinas[y, x] for y, x in coordenadas]
        indices_ordenados = np.argsort(fuerzas)[::-1]  # Descendente
        coordenadas = coordenadas[indices_ordenados[:num_semillas]]
    
    print(f"Detectadas {len(coordenadas)} semillas automáticamente")
    return [(y, x) for y, x in coordenadas]

def main():
    """
    Función principal para demostrar el crecimiento de regiones con GVF.
    """
    # --- Configuración ---
    image_filename = 'Cars129.png'
    
    # Construir la ruta a la imagen de forma robusta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, 'images', image_filename)
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
        print("Intentando cargar una imagen de ejemplo...")
        # Crear una imagen de ejemplo si no se encuentra el archivo
        image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.circle(image, (200, 200), 80, (255, 0, 0), -1)
        cv2.rectangle(image, (100, 100), (150, 150), (0, 255, 0), -1)
    
    # Parámetros del algoritmo
    parametros = {
        'umbral_gvf': 1.1,      # Umbral para el flujo del campo GVF
        'umbral_intensidad': 100, # Umbral de similitud de intensidad
        'tamano_maximo': 5000   # Tamaño máximo por región
    }
    
    # Semillas manuales (opcional - si no se proporcionan, se detectan automáticamente)
    semillas_manual = [
        (image.shape[0] // 2, image.shape[1] // 2),  # Centro
        (100, 100),  # Esquina superior izquierda
        (100, 300),  # Esquina superior derecha
    ]
    
    # Ejecutar segmentación con GVF
    gradiente_vis, gvf_vis, semillas_vis, mascara, resultado = gvf_region_growing_segmentation(
        image, semillas_manual=None, parametros=parametros  # Usar semillas automáticas
    )
    
    # Crear diccionario con las imágenes para mostrar
    images_dict = {
        '1. Imagen Original': image,
        '2. Gradiente (Jet)': gradiente_vis,
     #   '3. GVF (Jet)': gvf_vis,
     #   '4. Semillas Detectadas': semillas_vis,
        '5. Mascara Segmentacion': cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
     #   '6. Resultado Final': resultado
    }
    
    # Mostrar las imágenes ordenadas
    mostrar_imagenes_ordenadas(images_dict)
    
    print("Segmentación con GVF completada. Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar los resultados
    output_dir = os.path.join(project_root, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cv2.imwrite(os.path.join(output_dir, 'gvf_gradiente.png'), gradiente_vis)
    cv2.imwrite(os.path.join(output_dir, 'gvf_campo.png'), gvf_vis)
    cv2.imwrite(os.path.join(output_dir, 'gvf_semillas.png'), semillas_vis)
    cv2.imwrite(os.path.join(output_dir, 'gvf_mascara.png'), mascara)
    cv2.imwrite(os.path.join(output_dir, 'gvf_resultado.png'), resultado)
    
    print(f"Resultados guardados en la carpeta: {output_dir}")

if __name__ == "__main__":
    main()