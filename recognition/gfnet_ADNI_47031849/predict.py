"""
From Task Sheet:
"showing example usage of your trained model. Print out any results and / or provide visualisations
where applicable"
"""
import torch
import tqdm

def test_model(model, device):
    model.eval()

    predictions_list = []
    actual_values = []

    for images, labels in tqdm(test_dataset, disable=False, desc="Testing... "):
        # send the batch to the device
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)
        outputs = torch.sigmoid(outputs).mean().item()

        predictions = 1 if outputs >= 0.5 else 0
        predictions_list.append(predictions)
        actual_values.append(labels.item())

    return actual_values, predictions_list