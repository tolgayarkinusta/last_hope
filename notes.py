import pyzed

# depth_value = depth.get_value(center_x, center_y)[1]  # Sadece metre cinsinden değeri al


closest_depth = float('inf')
closest_center = None
for box in red_positions:
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    depth_val = depth.get_value(cx, cy)[1]
    if np.isnan(depth_val):
        continue
    if depth_val < closest_depth:
        closest_depth = depth_val
        closest_center = (cx, cy)
if closest_center is not None:
    cv2.line(frame, (center_x, center_y), closest_center, (0, 0, 255), thickness=2)
    if closest_depth < 0.89: