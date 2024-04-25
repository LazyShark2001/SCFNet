from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    model = YOLO('weight/SCFNet.pt')

    # Validate the model
    metrics = model.val(data='data.yaml')