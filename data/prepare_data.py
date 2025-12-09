import torch
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def save_csv(dataset, filename):
    images = dataset.data.reshape(len(dataset), -1).numpy()
    labels = dataset.targets.numpy()
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv(filename, index=False)

def main():
    os.makedirs("data", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    # Download Fashion-MNIST
    train_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    # Train/Val Split
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=10000, random_state=42
    )

    train_split = torch.utils.data.Subset(train_dataset, train_indices)
    val_split = torch.utils.data.Subset(train_dataset, val_indices)

    print("Saving CSV files...")

    save_csv(train_split, "data/fashion_mnist_train.csv")
    save_csv(val_split, "data/fashion_mnist_val.csv")
    save_csv(test_dataset, "data/fashion_mnist_test.csv")

    print("Done! CSV files saved in /data folder.")

if __name__ == "__main__":
    main()
