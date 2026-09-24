"""Perdidas y ponderaciones que los alumnos deben implementar."""

import numpy as np
import torch
import torch.nn.functional as F


def labels_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    TODO(alumno):
    Convierte clases enteras a umbrales binarios acumulativos.

    Ejemplo:
    Si num_classes = 5 y la etiqueta es 2, el vector debe ser [1, 1, 0, 0].

    Formas:
    - labels: (batch_size,)
    - salida: (batch_size, num_classes - 1)
    """
    labels = labels.long()#No choque de int 
    thresholds = torch.arange(num_classes - 1, device=labels.device)# k-1 fronteras y que se puedan procesar en el mismo espacio que los labels
    levels = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float() # compara cada etiqueta vs cada umbral: 1 si y > k, si no 0


    return levels

def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    BCE con logits sobre los K-1 umbrales ordinales.

    Formas:
    - logits: (batch_size, num_classes - 1)
    - labels: (batch_size,)
    - class_weights: (num_classes,) o None
    """
    levels = labels_to_levels(labels, num_classes)  # (B, K-1) ya implementado
    if class_weights is None:
        return F.binary_cross_entropy_with_logits(logits, levels)  # BCE con sigmoide 
    w = class_weights[labels].unsqueeze(1).expand_as(logits)  # peso por muestra replicado a (B, K-1)
    return F.binary_cross_entropy_with_logits(logits, levels, weight=w)  # promedio ponderado x muestra


def effective_number_weights(
    labels: np.ndarray,
    num_classes: int,
    beta: float = 0.99,
) -> torch.Tensor:
    """
    Pesos por numero efectivo de muestras:

        w_c = (1 - beta) / (1 - beta ** n_c)

    Normalizar los pesos para que su media sea 1.

    Formas:
    - labels: (N,)
    - salida: (num_classes,)
    """
    counts = np.bincount(labels, minlength=num_classes)  # n_c por clase
    n_c = torch.tensor(counts, dtype=torch.float32)  # No choque de int
    # Evita división por cero si alguna clase no aparece
    w = torch.where(
        n_c > 0,
        (1.0 - beta) / (1.0 - beta ** n_c),
        torch.zeros_like(n_c)
    )
    w = w / w.mean()  # Media = 1
    return w

    
