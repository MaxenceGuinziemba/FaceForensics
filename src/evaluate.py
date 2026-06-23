"""
Script d'évaluation pour la détection de deepfakes.

Charge un modèle entraîné et calcule les métriques détaillées sur le test set :
accuracy, precision, recall, F1, AUC-ROC, matrice de confusion,
et performances par méthode de manipulation.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model.pth --data_root data --compression c40
"""
import argparse
import json
import os
from os.path import join

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.models import model_selection
from src.data import FaceForensicsDataset, xception_default_data_transforms
from src.data.dataset import METHODS


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Passe tout le dataset dans le modèle et collecte :
    - les vrais labels
    - les labels prédits
    - les probabilités (softmax) pour l'AUC
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    softmax = nn.Softmax(dim=1)

    for frames, labels in tqdm(loader, desc='  Évaluation'):
        frames = frames.to(device)
        outputs = model(frames)
        probs = softmax(outputs)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # probabilité de "fake"

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, output_path):
    """Sauvegarde la matrice de confusion en image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')

    labels = ['Real', 'Fake']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('Matrice de confusion')

    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, output_path):
    """Sauvegarde la courbe ROC en image."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbe ROC')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_per_method(per_method_data, output_path):
    """Courbes ROC superposées pour chaque méthode de manipulation."""
    colors = {'Deepfakes': '#e74c3c', 'Face2Face': '#3498db',
              'FaceSwap': '#2ecc71', 'NeuralTextures': '#9b59b6'}

    fig, ax = plt.subplots(figsize=(7, 6))

    for method, data in per_method_data.items():
        if 'labels' not in data or len(np.unique(data['labels'])) < 2:
            continue
        fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
        auc = roc_auc_score(data['labels'], data['probs'])
        color = colors.get(method, None)
        ax.plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})', color=color)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbes ROC par méthode de manipulation')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_example_predictions(model, loader, device, output_path):
    """Grille d'exemples : vrais/faux positifs et négatifs avec probabilité."""
    model.eval()
    softmax = nn.Softmax(dim=1)
    label_names = ['Real', 'Fake']

    categories = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    max_per_cat = 2

    with torch.no_grad():
        for frames, labels in loader:
            frames_dev = frames.to(device)
            outputs = model(frames_dev)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            for i in range(frames.size(0)):
                true_l = labels[i].item()
                pred_l = preds[i].item()
                prob = probs[i, pred_l].item()
                img = frames[i]

                if true_l == 1 and pred_l == 1:
                    cat = 'TP'
                elif true_l == 0 and pred_l == 0:
                    cat = 'TN'
                elif true_l == 0 and pred_l == 1:
                    cat = 'FP'
                else:
                    cat = 'FN'

                if len(categories[cat]) < max_per_cat:
                    categories[cat].append((img, true_l, pred_l, prob))

            if all(len(v) >= max_per_cat for v in categories.values()):
                break

    all_examples = []
    titles = []
    cat_labels = {'TP': 'Vrai Positif', 'TN': 'Vrai Négatif',
                  'FP': 'Faux Positif', 'FN': 'Faux Négatif'}
    for cat in ['TN', 'TP', 'FN', 'FP']:
        for img, true_l, pred_l, prob in categories[cat]:
            all_examples.append(img)
            titles.append(f'{cat_labels[cat]}\n{label_names[true_l]}→{label_names[pred_l]} ({prob:.2f})')

    if not all_examples:
        return

    n = len(all_examples)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, (img, title) in enumerate(zip(all_examples, titles)):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_per_method(model, data_root, split_path, compression,
                        transform, device, num_frames, batch_size, faces_dir=None):
    """
    Évalue le modèle séparément pour chaque méthode de manipulation.
    Permet de savoir si le modèle détecte mieux Deepfakes que NeuralTextures par exemple.
    """
    results = {}

    for method in METHODS:
        dataset = FaceForensicsDataset(
            data_root=data_root,
            split_path=split_path,
            compression=compression,
            methods=[method],
            transform=transform,
            num_frames_per_video=num_frames,
            faces_dir=faces_dir,
        )

        if len(dataset.samples) == 0:
            results[method] = {'status': 'pas de données'}
            continue

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        labels, preds, probs = collect_predictions(model, loader, device)

        if len(np.unique(labels)) < 2:
            results[method] = {
                'samples': len(labels),
                'accuracy': accuracy_score(labels, preds),
                'status': 'une seule classe présente, AUC non calculable',
            }
            continue

        results[method] = {
            'samples': len(labels),
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auc': roc_auc_score(labels, probs),
            'labels': labels,
            'probs': probs,
        }

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---------------------------------------------------------------
    # Charger le checkpoint
    # ---------------------------------------------------------------
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    model_name = saved_args.get('model', args.model)
    dropout = saved_args.get('dropout', args.dropout)

    print(f'Checkpoint: {args.checkpoint}')
    print(f'  Epoch: {checkpoint.get("epoch", "?")}')
    print(f'  Val accuracy: {checkpoint.get("val_acc", "?"):.4f}')
    print(f'  Modèle: {model_name} (dropout={dropout})')

    # ---------------------------------------------------------------
    # Recréer le modèle et charger les poids entraînés
    # ---------------------------------------------------------------
    model, image_size, *_ = model_selection(
        modelname=model_name,
        num_out_classes=2,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # ---------------------------------------------------------------
    # Dataset de test
    # ---------------------------------------------------------------
    split_path = join(args.splits_dir, 'test.json')
    test_dataset = FaceForensicsDataset(
        data_root=args.data_root,
        split_path=split_path,
        compression=args.compression,
        transform=xception_default_data_transforms['test'],
        num_frames_per_video=args.frames_per_video,
        faces_dir=args.faces_dir,
    )

    if len(test_dataset.samples) == 0:
        # Fallback sur val si pas de données test (cas de notre échantillon local)
        print('Aucune vidéo test trouvée, fallback sur le val set')
        split_path = join(args.splits_dir, 'val.json')
        test_dataset = FaceForensicsDataset(
            data_root=args.data_root,
            split_path=split_path,
            compression=args.compression,
            transform=xception_default_data_transforms['val'],
            num_frames_per_video=args.frames_per_video,
            faces_dir=args.faces_dir,
        )

    counts = test_dataset.get_label_counts()
    print(f'Évaluation: {len(test_dataset.samples)} vidéos ({counts}) → {len(test_dataset)} samples')

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------------
    # Collecter toutes les prédictions
    # ---------------------------------------------------------------
    print('\n--- Métriques globales ---')
    labels, preds, probs = collect_predictions(model, loader, device)

    # ---------------------------------------------------------------
    # Métriques globales
    # ---------------------------------------------------------------
    results = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

    if len(np.unique(labels)) >= 2:
        results['auc'] = roc_auc_score(labels, probs)

    for metric, value in results.items():
        print(f'  {metric:12s}: {value:.4f}')

    # Classification report détaillé
    print('\n--- Rapport détaillé ---')
    print(classification_report(labels, preds, target_names=['Real', 'Fake'], zero_division=0))

    # ---------------------------------------------------------------
    # Matrice de confusion
    # ---------------------------------------------------------------
    cm = confusion_matrix(labels, preds)
    print('Matrice de confusion:')
    print(f'  Real prédit Real:  {cm[0][0]:4d}  |  Real prédit Fake:  {cm[0][1]:4d}')
    print(f'  Fake prédit Real:  {cm[1][0]:4d}  |  Fake prédit Fake:  {cm[1][1]:4d}')

    # ---------------------------------------------------------------
    # Graphiques
    # ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    cm_path = join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)
    print(f'\nMatrice de confusion sauvegardée: {cm_path}')

    if 'auc' in results:
        roc_path = join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(labels, probs, roc_path)
        print(f'Courbe ROC sauvegardée: {roc_path}')

    examples_path = join(args.output_dir, 'example_predictions.png')
    plot_example_predictions(model, loader, device, examples_path)
    print(f'Exemples de prédictions sauvegardés: {examples_path}')

    # ---------------------------------------------------------------
    # Évaluation par méthode
    # ---------------------------------------------------------------
    print('\n--- Performances par méthode ---')
    per_method = evaluate_per_method(
        model, args.data_root, split_path, args.compression,
        xception_default_data_transforms['test'],
        device, args.frames_per_video, args.batch_size,
        faces_dir=args.faces_dir,
    )

    for method, metrics in per_method.items():
        if 'accuracy' in metrics:
            line = f'  {method:16s}: acc={metrics["accuracy"]:.4f}'
            if 'auc' in metrics:
                line += f'  auc={metrics["auc"]:.4f}'
                line += f'  f1={metrics["f1"]:.4f}'
            if 'status' in metrics:
                line += f'  ({metrics["status"]})'
            print(line)
        else:
            print(f'  {method:16s}: {metrics.get("status", "inconnu")}')

    # ---------------------------------------------------------------
    # Courbes ROC par méthode
    # ---------------------------------------------------------------
    roc_methods_path = join(args.output_dir, 'roc_curves_per_method.png')
    plot_roc_per_method(per_method, roc_methods_path)
    print(f'Courbes ROC par méthode sauvegardées: {roc_methods_path}')

    # ---------------------------------------------------------------
    # Sauvegarder le rapport complet en JSON
    # ---------------------------------------------------------------
    report_per_method = {}
    for method, metrics in per_method.items():
        report_per_method[method] = {k: v for k, v in metrics.items()
                                      if k not in ('labels', 'probs')}

    report = {
        'checkpoint': args.checkpoint,
        'compression': args.compression,
        'num_samples': len(labels),
        'global_metrics': results,
        'per_method': report_per_method,
        'confusion_matrix': cm.tolist(),
    }

    report_path = join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nRapport complet sauvegardé: {report_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Évaluation du modèle de détection de deepfakes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Chemin vers le dossier data/')
    parser.add_argument('--splits_dir', type=str, default='configs/splits',
                        help='Dossier contenant les fichiers split JSON')
    parser.add_argument('--compression', type=str, default='c40',
                        choices=['c0', 'c23', 'c40'],
                        help='Niveau de compression')
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet18'],
                        help='Architecture (utilisé seulement si pas dans le checkpoint)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout (utilisé seulement si pas dans le checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--faces_dir', type=str, default=None,
                        help='Dossier des visages pré-extraits')
    parser.add_argument('--frames_per_video', type=int, default=10,
                        help='Frames par vidéo pour l\'évaluation')
    parser.add_argument('--output_dir', type=str, default='checkpoints/evaluation',
                        help='Dossier pour les résultats (graphiques, rapport)')

    args = parser.parse_args()
    main(args)
