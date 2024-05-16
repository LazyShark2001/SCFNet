from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('module/SCFNet.yaml')

    model.train(data="data.yaml", epochs=400, batch=32, patience=0)