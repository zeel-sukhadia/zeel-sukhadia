from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best_OBB.pt')

# Run inference on an image
results = model('./images/stallion.jpeg', show=True)  # results list

# View results

for r in results:
    obb_cls = r.obb.cls
    obb_xywhr = r.obb.xywhr
    obb_xyxy = r.obb.xyxy

    for label, coords_xywhr, coords_xyxy in zip(obb_cls, obb_xywhr, obb_xyxy):
        class_label = model.names[int(label)]
        x_center, y_center, width, height, rotation = coords_xywhr
        x1, y1, x2, y2 = coords_xyxy
        print(f'Label: {class_label}')
        print(f'   Coordinates (xywhr): ({x_center}, {y_center}), Width: {width}, Height: {height}, Rotation: {rotation}')
        print(f'   Coordinates (xyxy): ({x1}, {y1}), ({x2}, {y2})')
