import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score
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

def cargar_y_preprocesar_imagen(ruta_imagen, es_ground_truth=False):
    """
    Carga y preprocesa una imagen para asegurar que sea binaria (0 y 255)
    """
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
    
    # Umbralizar para asegurar que sea binaria
    _, imagen_binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    
    return imagen_binaria

def calcular_iou(mascara_pred, mascara_gt):
    """
    Calcula el Intersection over Union (IoU)
    """
    # Convertir a booleanos para el cálculo
    pred_bool = (mascara_pred > 0)
    gt_bool = (mascara_gt > 0)
    
    # Calcular intersección y unión
    interseccion = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    
    # Evitar división por cero
    if union == 0:
        return 0.0
    
    iou = interseccion / union
    return iou

def calcular_dice(mascara_pred, mascara_gt):
    """
    Calcula el Dice Coefficient
    """
    # Convertir a booleanos
    pred_bool = (mascara_pred > 0)
    gt_bool = (mascara_gt > 0)
    
    # Calcular intersección y suma de áreas
    interseccion = np.logical_and(pred_bool, gt_bool).sum()
    area_pred = pred_bool.sum()
    area_gt = gt_bool.sum()
    
    # Evitar división por cero
    if (area_pred + area_gt) == 0:
        return 1.0  # Ambas máscaras están vacías
    
    dice = (2 * interseccion) / (area_pred + area_gt)
    return dice

def calcular_precision_recall(mascara_pred, mascara_gt):
    """
    Calcula Precision y Recall
    """
    # Aplanar las máscaras para usar con sklearn
    pred_flat = (mascara_pred > 0).flatten()
    gt_flat = (mascara_gt > 0).flatten()
    
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    return precision, recall

def crear_imagenes_comparacion(mascara_pred, mascara_gt, iou, dice, precision, recall, f1_score):
    """
    Crea las imágenes individuales para la comparación
    """
    imagenes = {}
    
    # 1. Máscara predicha (original)
    mascara_pred_color = cv2.cvtColor(mascara_pred, cv2.COLOR_GRAY2BGR)
    cv2.putText(mascara_pred_color, "Mascara Predicha", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    imagenes["1. Predicha"] = mascara_pred_color
    
    # 2. Ground Truth (original)
    mascara_gt_color = cv2.cvtColor(mascara_gt, cv2.COLOR_GRAY2BGR)
    cv2.putText(mascara_gt_color, "Ground Truth", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    imagenes["2. Ground Truth"] = mascara_gt_color
    
    # 3. Superposición (ambas máscaras)
    superposicion = cv2.addWeighted(mascara_pred_color, 0.5, mascara_gt_color, 0.5, 0)
    cv2.putText(superposicion, "Superposicion (50/50)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    imagenes["3. Superposicion"] = superposicion
    
    # 4. Comparación detallada (verde, rojo, azul)
    comparacion = np.zeros((mascara_pred.shape[0], mascara_pred.shape[1], 3), dtype=np.uint8)
    
    # Coincidencias (verde)
    coincidencias = np.logical_and(mascara_pred > 0, mascara_gt > 0)
    comparacion[coincidencias] = [0, 255, 0]  # Verde en BGR
    
    # Falsos positivos (rojo) - predicho pero no en GT
    falsos_positivos = np.logical_and(mascara_pred > 0, mascara_gt == 0)
    comparacion[falsos_positivos] = [0, 0, 255]  # Rojo en BGR
    
    # Falsos negativos (azul) - en GT pero no predicho
    falsos_negativos = np.logical_and(mascara_pred == 0, mascara_gt > 0)
    comparacion[falsos_negativos] = [255, 0, 0]  # Azul en BGR
    
    cv2.putText(comparacion, "Comparacion Detallada", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparacion, "Verde: Coincidencias (TP)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(comparacion, "Rojo: Falsos Positivos (FP)", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(comparacion, "Azul: Falsos Negativos (FN)", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    imagenes["4. Comparacion"] = comparacion
    
    # 5. Panel de métricas
    altura, ancho = mascara_pred.shape
    panel_metricas = np.zeros((altura, ancho, 3), dtype=np.uint8)
    
    # Calcular estadísticas
    pred_pixels = (mascara_pred > 0).sum()
    gt_pixels = (mascara_gt > 0).sum()
    interseccion = np.logical_and(mascara_pred > 0, mascara_gt > 0).sum()
    
    # Mostrar métricas en el panel
    cv2.putText(panel_metricas, "METRICAS DE EVALUACION", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.putText(panel_metricas, f"IoU: {iou:.4f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel_metricas, f"Dice: {dice:.4f}", (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel_metricas, f"Precision: {precision:.4f}", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel_metricas, f"Recall: {recall:.4f}", (20, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel_metricas, f"F1-Score: {f1_score:.4f}", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(panel_metricas, "ESTADISTICAS:", (20, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(panel_metricas, f"Pixeles Pred: {pred_pixels}", (20, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel_metricas, f"Pixeles GT: {gt_pixels}", (20, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel_metricas, f"Interseccion: {interseccion}", (20, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    imagenes["5. Metricas"] = panel_metricas
    
    return imagenes

def evaluar_mascaras(ruta_predicha, ruta_ground_truth, mostrar_visualizacion=True):
    """
    Función principal para evaluar las máscaras
    """
    print("=== EVALUACIÓN DE MÁSCARAS ===")
    print(f"Máscara predicha: {os.path.basename(ruta_predicha)}")
    print(f"Ground truth: {os.path.basename(ruta_ground_truth)}")
    print()
    
    try:
        # Cargar y preprocesar imágenes
        mascara_pred = cargar_y_preprocesar_imagen(ruta_predicha, es_ground_truth=False)
        mascara_gt = cargar_y_preprocesar_imagen(ruta_ground_truth, es_ground_truth=True)
        
        # Verificar que las máscaras tengan el mismo tamaño
        if mascara_pred.shape != mascara_gt.shape:
            print("¡Advertencia: Las máscaras tienen tamaños diferentes!")
            print(f"Tamaño predicha: {mascara_pred.shape}")
            print(f"Tamaño GT: {mascara_gt.shape}")
            
            # Redimensionar la máscara predicha al tamaño del GT
            mascara_pred = cv2.resize(mascara_pred, (mascara_gt.shape[1], mascara_gt.shape[0]))
            print("Máscara predicha redimensionada para coincidir con GT")
        
        # Calcular métricas
        iou = calcular_iou(mascara_pred, mascara_gt)
        dice = calcular_dice(mascara_pred, mascara_gt)
        precision, recall = calcular_precision_recall(mascara_pred, mascara_gt)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mostrar resultados en consola
        print("=== RESULTADOS ===")
        print(f"IoU (Intersection over Union): {iou:.4f}")
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print()
        
        # Estadísticas adicionales
        pred_pixels = (mascara_pred > 0).sum()
        gt_pixels = (mascara_gt > 0).sum()
        interseccion = np.logical_and(mascara_pred > 0, mascara_gt > 0).sum()
        
        print("=== ESTADÍSTICAS DETALLADAS ===")
        print(f"Píxeles en máscara predicha: {pred_pixels}")
        print(f"Píxeles en ground truth: {gt_pixels}")
        print(f"Píxeles en intersección: {interseccion}")
        
        if gt_pixels > 0:
            print(f"Porcentaje de cobertura: {(pred_pixels/gt_pixels*100):.2f}%")
        
        # Crear y mostrar visualización si se solicita
        if mostrar_visualizacion:
            imagenes = crear_imagenes_comparacion(mascara_pred, mascara_gt, iou, dice, precision, recall, f1_score)
            mostrar_imagenes_ordenadas(imagenes)
            
            print("\nPresiona cualquier tecla en una de las ventanas para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'pred_pixels': pred_pixels,
            'gt_pixels': gt_pixels,
            'intersection': interseccion
        }
        
    except Exception as e:
        print(f"Error al procesar las imágenes: {e}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    # Rutas a las imágenes
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    ruta_predicha = os.path.join(output_dir, "Cars11_watershed_evaluar.png")
    ruta_ground_truth = os.path.join(output_dir, "Cars11_GT.png")
    
    # Verificar que los archivos existan
    if not os.path.exists(ruta_predicha):
        print(f"Error: No se encuentra el archivo {ruta_predicha}")
        print("Por favor, asegúrate de que el archivo existe en el directorio actual.")
    elif not os.path.exists(ruta_ground_truth):
        print(f"Error: No se encuentra el archivo {ruta_ground_truth}")
        print("Por favor, asegúrate de que el archivo existe en el directorio actual.")
    else:
        # Ejecutar evaluación
        resultados = evaluar_mascaras(ruta_predicha, ruta_ground_truth, mostrar_visualizacion=True)
        
        # Guardar resultados en un archivo de texto
        if resultados is not None:
            with open("resultados_evaluacion.txt", "w") as f:
                f.write("RESULTADOS DE EVALUACIÓN DE MÁSCARAS\n")
                f.write("===================================\n\n")
                f.write(f"Máscara predicha: {ruta_predicha}\n")
                f.write(f"Ground truth: {ruta_ground_truth}\n\n")
                f.write("MÉTRICAS:\n")
                f.write(f"IoU: {resultados['iou']:.4f}\n")
                f.write(f"Dice Coefficient: {resultados['dice']:.4f}\n")
                f.write(f"Precision: {resultados['precision']:.4f}\n")
                f.write(f"Recall: {resultados['recall']:.4f}\n")
                f.write(f"F1-Score: {resultados['f1_score']:.4f}\n\n")
                f.write("ESTADÍSTICAS:\n")
                f.write(f"Píxeles en máscara predicha: {resultados['pred_pixels']}\n")
                f.write(f"Píxeles en ground truth: {resultados['gt_pixels']}\n")
                f.write(f"Píxeles en intersección: {resultados['intersection']}\n")
            
            print("\nResultados guardados en 'resultados_evaluacion.txt'")