classes = []

for idx in range(len(folders)): classes.append(folders[idx])

truelabels = []
predictions = []

with torch.no_grad():
    model.eval()

    n_correct = 0
    n_samples = 0
    n_class_correct_pred = {classname: 0 for classname in classes}
    n_class_total_pred = {classname: 0 for classname in classes}

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs,1)
     
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()

        for label, prediction in zip(labels, predicted):
            truelabels.append(label.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            #print(f"label->{label}------ prediction->{prediction}")
            #print(f"labeltype->{type(label)}             pred type->{type(prediction)}")
            if label == prediction:
                n_class_correct_pred[classes[label]] += 1
            n_class_total_pred[classes[label]] += 1
        
    #test_loss_lst.append(test_loss/len(test_loader))

    acc =  100.0 * n_correct/n_samples
    print(f'Accuracy of the network: {acc:.5f} %\n')

    for classname, correct_count in n_class_correct_pred.items():
        acc = 100.0 * float(correct_count)/n_class_total_pred[classname]
        print(f'Accuracy of {classname:s}: {acc:.3f} %')
print(n_class_correct_pred)
