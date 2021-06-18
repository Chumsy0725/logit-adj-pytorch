from model.config import *
from model.transforms import *


def _train(train_loader):
    total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % total_steps == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}")

        # Decay Learning Rate
        lr_scheduler.step()


def _save(path):
    torch.save(model.state_dict(), path)


def _test(test_loader, classes):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(test_batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network: {acc} %")

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {acc} %")


def run(dataset, root):
    train_dataset = dataset(root=root,
                            train=True,
                            transform=TRAIN_TRANSFORMS[dataset.get_identifier()],
                            download=True)

    test_dataset = dataset(root=root,
                           train=False,
                           transform=TEST_TRANSFORMS[dataset.get_identifier()])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    # Train the model
    _train(train_loader)
    print("Finished Training")

    # Save the Model.
    _save(model_save_path + "/" + dataset.get_identifier() + "_trained_" + str(num_epochs) + ".pth")

    # Test the model for accuracy.
    _test(test_loader, test_dataset.get_classes())
