# train함수
from data_loader import *
from tqdm.auto import tqdm
# from utils import calculate_accuracy, update_confusion_matrix, calculate_precision_recall,conf_to_df,plot_confusion_heatmap, calculate_and_plot_f1_scores,plot_loss,plot_acc,image_transform
from utils import *



def train(model, train_loader, optimizer, criterion, device,num_classes):
    model.train()
    train_loss = 0.0
    train_correct = 0
    conf_matrix_train = torch.zeros(num_classes, num_classes)

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += calculate_accuracy(outputs, targets).item()
        conf_matrix_train = update_confusion_matrix(conf_matrix_train, outputs, targets)

    average_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / len(train_loader)

    return average_train_loss, train_accuracy, conf_matrix_train

#validation 함수
def validate(model, valid_loader, criterion, device,num_classes):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    conf_matrix_val = torch.zeros(num_classes, num_classes)
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_correct += calculate_accuracy(outputs, targets).item()
            conf_matrix_val = update_confusion_matrix(conf_matrix_val, outputs, targets)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)

    average_val_loss = val_loss / len(valid_loader)
    val_accuracy = val_correct / len(valid_loader)

    return average_val_loss, val_accuracy, conf_matrix_val, all_targets, all_outputs
