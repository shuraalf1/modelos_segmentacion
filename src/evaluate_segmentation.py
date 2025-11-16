import cv2
import numpy as np
import os

def calculate_metrics(ground_truth, prediction):
    """
    Calcula las métricas de segmentación (Precisión, Recall, IoU, F1-Score).

    :param ground_truth: Máscara binaria de la verdad fundamental.
    :param prediction: Máscara binaria de la predicción del algoritmo.
    :return: Un diccionario con las métricas calculadas.
    """
    # Binarizar las máscaras para asegurar que solo contengan 0 y 255
    _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

    # Asegurarse de que las máscaras sean booleanas
    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    # Calcular Verdaderos Positivos, Falsos Positivos y Falsos Negativos
    tp = np.sum(np.logical_and(pred, gt))  # True Positives
    fp = np.sum(np.logical_and(pred, np.logical_not(gt)))  # False Positives
    fn = np.sum(np.logical_and(np.logical_not(pred), gt))  # False Negatives

    # Calcular Métricas (manejando divisiones por cero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "F1 Score": f1_score
    }

def main():
    # --- Configuración de las imágenes de entrada (máscaras) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    # Asegúrate de que estas rutas apunten a tus archivos de máscaras
    # Puedes usar máscaras generadas por tus otros scripts o creadas manualmente.
    path_real = os.path.join(project_root, 'output', 'real_a.png') # CAMBIA ESTO por tu máscara real
    path_a = os.path.join(project_root, 'output', 'result_watershed_a.png') # CAMBIA ESTO por la máscara A
    path_b = os.path.join(project_root, 'output', 'result_growing_a.png') # CAMBIA ESTO por la máscara B

    # 1. Cargar las 3 máscaras como imágenes en escala de grises
    mask_real = cv2.imread(path_real, cv2.IMREAD_GRAYSCALE)
    mask_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    # Verificar que las imágenes se cargaron correctamente
    if mask_real is None:
        print(f"Error: No se pudo cargar la máscara REAL desde: {path_real}")
        return
    if mask_a is None:
        print(f"Error: No se pudo cargar la máscara A desde: {path_a}")
        return
    if mask_b is None:
        print(f"Error: No se pudo cargar la máscara B desde: {path_b}")
        return
    
    print("Máscaras cargadas exitosamente.")

    # 2. Calcular métricas para cada predicción contra la máscara real
    print("Calculando métricas para la Máscara A...")
    metrics_a = calculate_metrics(mask_real, mask_a)
    
    print("Calculando métricas para la Máscara B...")
    metrics_b = calculate_metrics(mask_real, mask_b)

    # 3. Mostrar resultados en una tabla
    print("\n\n--- TABLA DE RESULTADOS DE EVALUACIÓN ---")
    print("-" * 60)
    print(f"{'Métrica':<15} | {'Máscara A':<20} | {'Máscara B':<20}")
    print("-" * 60)
    for metric_name in metrics_a.keys():
        val_a = f"{metrics_a[metric_name]:.4f}"
        val_b = f"{metrics_b[metric_name]:.4f}"
        print(f"{metric_name:<15} | {val_a:<20} | {val_b:<20}")
    print("-" * 60)

    # Visualización opcional para comprobar las máscaras
    cv2.imshow('Mascara Real (Ground Truth)', mask_real)
    cv2.imshow('Mascara A', mask_a)
    cv2.imshow('Mascara B', mask_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()