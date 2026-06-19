"""
Script d'entraînement pour la détection de deepfakes avec XceptionNet.

Usage:
    python -m src.train --data_root data --compression c40 --epochs 5       # test local
    python -m src.train --data_root ~/datasets/ff++ --compression c23       # cluster GPU

Tout est configurable via les arguments en ligne de commande (voir --help).
"""
import argparse
import os
import time
from os.path import join

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import model_selection
from src.data import FaceForensicsDataset, xception_default_data_transforms


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss par epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy par epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Une epoch d'entraînement.
    Retourne la loss moyenne et l'accuracy sur tout le train set.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(loader, desc='  Train', leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        # Forward : le modèle prédit
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward : on ajuste les poids
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Métriques
        running_loss += loss.item() * frames.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Évaluation sur le val set (pas de backpropagation).
    Retourne la loss moyenne et l'accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(loader, desc='  Val', leave=False):
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * frames.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main(args):
    # ---------------------------------------------------------------
    # Device : GPU si disponible, sinon CPU
    # ---------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---------------------------------------------------------------
    # Modèle : XceptionNet adapté pour 2 classes (real/fake)
    # ---------------------------------------------------------------
    model, image_size, *_ = model_selection(
        modelname=args.model,
        num_out_classes=2,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(f'Modèle: {args.model} (dropout={args.dropout})')

    # ---------------------------------------------------------------
    # Datasets : train et val
    # ---------------------------------------------------------------
    train_dataset = FaceForensicsDataset(
        data_root=args.data_root,
        split_path=join(args.splits_dir, 'train.json'),
        compression=args.compression,
        transform=xception_default_data_transforms['train'],
        num_frames_per_video=args.frames_per_video,
    )
    val_dataset = FaceForensicsDataset(
        data_root=args.data_root,
        split_path=join(args.splits_dir, 'val.json'),
        compression=args.compression,
        transform=xception_default_data_transforms['val'],
        num_frames_per_video=args.frames_per_video,
    )

    print(f'Train: {len(train_dataset.samples)} vidéos ({train_dataset.get_label_counts()}) → {len(train_dataset)} samples/epoch')
    print(f'Val:   {len(val_dataset.samples)} vidéos ({val_dataset.get_label_counts()}) → {len(val_dataset)} samples/epoch')

    # ---------------------------------------------------------------
    # DataLoaders : regroupent les images en batch
    # ---------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ---------------------------------------------------------------
    # CrossEntropyLoss avec Label Smoothing : réduit l'overfitting
    # label_smoothing=0.1 : transforme 0→0.05, 1→0.95 au lieu de 0→0, 1→1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adam : algorithme d'optimisation qui ajuste les poids du modèle
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler : Cosine Annealing with Warm Restarts
    # Plus stable que ReduceLROnPlateau, redémarre périodiquement
    # T_0=10 : période initiale de 10 epochs
    # T_mult=2 : doubler la période à chaque redémarrage
    # eta_min=1e-6 : LR minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # ---------------------------------------------------------------
    # TensorBoard : pour visualiser les courbes d'entraînement
    # ---------------------------------------------------------------
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---------------------------------------------------------------
    # Boucle d'entraînement
    # ---------------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Warmup LR : augmenter progressivement le LR pendant les 3 premières epochs
    warmup_epochs = 3
    warmup_lr_start = 1e-6

    print(f'\nDémarrage: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}')
    print(f'Warmup: {warmup_epochs} epochs ({warmup_lr_start:.1e} → {args.lr:.1e})')
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        # Warmup : augmenter progressivement le LR
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            warmup_lr = warmup_lr_start + (args.lr - warmup_lr_start) * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler : step après le warmup
        if epoch > warmup_epochs:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # Affichage
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train loss: {train_loss:.4f} acc: {train_acc:.4f} | '
              f'Val loss: {val_loss:.4f} acc: {val_acc:.4f} | '
              f'lr: {current_lr:.1e} | '
              f'{elapsed:.0f}s')

        # TensorBoard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint_path = join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f'  → Nouveau meilleur modèle sauvegardé (val_acc={val_acc:.4f})')
        else:
            epochs_without_improvement += 1

        # Early stopping : arrêter si pas d'amélioration depuis N epochs
        if epochs_without_improvement >= args.patience:
            print(f'\nEarly stopping: pas d\'amélioration depuis {args.patience} epochs')
            break

    writer.close()

    curves_path = join(args.checkpoint_dir, 'training_curves.png')
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        curves_path,
    )
    print(f'\nCourbes d\'entraînement sauvegardées: {curves_path}')
    print(f'Meilleure val accuracy: {best_val_acc:.4f}')
    print(f'Modèle sauvegardé dans: {args.checkpoint_dir}/best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entraînement détection de deepfakes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Données
    parser.add_argument('--data_root', type=str, default='data',
                        help='Chemin vers le dossier data/')
    parser.add_argument('--splits_dir', type=str, default='configs/splits',
                        help='Dossier contenant train.json, val.json, test.json')
    parser.add_argument('--compression', type=str, default='c40',
                        choices=['c0', 'c23', 'c40'],
                        help='Niveau de compression des vidéos')

    # Modèle
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'resnet18'],
                        help='Architecture du modèle')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Taux de dropout (0 = pas de dropout)')

    # Entraînement
    parser.add_argument('--epochs', type=int, default=50,
                        help='Nombre maximum d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch (réduire si Out Of Memory)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate initial')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Régularisation L2')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping: epochs sans amélioration avant arrêt')
    parser.add_argument('--frames_per_video', type=int, default=10,
                        help='Nombre de frames échantillonnées par vidéo par epoch')

    # Workers
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Nombre de workers pour le chargement des données')

    # Sorties
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Dossier pour sauvegarder les modèles')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Dossier pour les logs TensorBoard')

    args = parser.parse_args()
    main(args)
