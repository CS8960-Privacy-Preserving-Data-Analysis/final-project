
import os
import matplotlib.pyplot as plt


def plot_train_test_loss_accuracy_vs_epochs(epochs, train_loss, test_loss, train_acc, test_acc, epsilon_value, save_dir):
    # Create the folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot for train/test loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', color='orange', linestyle='--', marker='x')
    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss plot
    loss_plot_path = os.path.join(save_dir, "train_test_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot for train/test accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, test_acc, label='Test Accuracy', color='orange', linestyle='--', marker='x')
    plt.title(f'Train and Test Accuracy Over Epochs (ε = {epsilon_value})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the accuracy plot
    acc_plot_path = os.path.join(save_dir, "train_test_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close()

    print(f"Plots saved in directory: {save_dir}")


def plot_train_test_loss_accuracy_vs_epsilon(epsilons, train_losses, test_losses, train_accuracies, test_accuracies, save_dir):
    # Create the folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot for combined train/test loss vs epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epsilons, test_losses, label='Test Loss', color='orange', linestyle='--', marker='x')
    plt.title('Train and Test Loss vs Privacy Budget (ε)')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the combined loss plot
    loss_plot_path = os.path.join(save_dir, "train_test_loss_vs_epsilon.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot for combined train/test accuracy vs epsilon
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, train_accuracies, label='Train Accuracy', color='blue', linestyle='-', marker='o')
    plt.plot(epsilons, test_accuracies, label='Test Accuracy', color='orange', linestyle='--', marker='x')
    plt.title('Train and Test Accuracy vs Privacy Budget (ε)')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the combined accuracy plot
    acc_plot_path = os.path.join(save_dir, "train_test_accuracy_vs_epsilon.png")
    plt.savefig(acc_plot_path)
    plt.close()

    print(f"Plots saved in directory: {save_dir}")