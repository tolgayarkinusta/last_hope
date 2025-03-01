def main():
    global width, manual_mode, magnetic_heading
    print("Initializing Camera...")
    try:
        zed = initialize_camera()
        print("Camera initialized! Initializing positional tracking...")
        initialize_positional_tracking(zed)
        print("Tracking initialized! Initializing spatial mapping...")
        initialize_spatial_mapping(zed)
        print("Mapping initialized!")

        # Kamera çözünürlüğünü al
        camera_info = zed.get_camera_information()
        width = camera_info.camera_configuration.resolution.width
        print(width)
        height = camera_info.camera_configuration.resolution.height
        print(height)
        # Görüntüde merkez noktasını hesapla
        center_x = width // 2
        center_y = height // 2
        print("Kamera çözünürlüğü: ", width, "x", height)
        print("Görüntü orta noktası: ", (center_x, center_y))

        # Used to store the sensors timestamp to know if the sensors_data is a new one or not
        ts_handler = TimestampHandler()

        # Görüntü ve derinlik verilerini almak için Mat nesneleri oluştur
        image = sl.Mat()
        depth = sl.Mat()
        pose = sl.Pose()
        # mesh = sl.Mesh()

        # Sensör verisi al
        sensors_data = sl.SensorsData()

        translation = pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
        start_x = translation[0]
        start_y = -(translation[1])
        print("Başlangıç konumu kaydedildi:", start_x, start_y)

        manual_mode = False

        # --- MISSION CONSTANTS ---
        MISSION_AVOID_BUOYS = 1
        MISSION_DOCKING = 2  # You can add more intermediate missions later
        MISSION_SPEED_CHALLENGE = 3
        MISSION_RETURN_HOME = 4

        MISSION_SPECIAL = 99    #ball and water delivery

        mission_state = MISSION_AVOID_BUOYS
        prev_mission_state = mission_state

        # Sonsuz bir döngüde görüntü akışı
        while True:
            # Kameradan bir yeni kare alın
            if zed.grab() == sl.ERROR_CODE.SUCCESS:

                # Görüntü ve derinlik verilerini al
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # OpenCV formatına dönüştür
                frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)  # BGRA -> BGR
                results = model(frame, conf=0.45)[0]

                # yolo sonuçlarının sv.Detections formatına dönüştürülmesi
                detections = sv.Detections.from_ultralytics(results)

                # tespitlerin sınırlarının ve etiketlerinin oluşturulması
                frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections)

                # tespitlerin koordinatlarının sınıflarının alınması
                coordinates = detections.xyxy.tolist()
                class_ids = detections.class_id.tolist()

                # retrieve the current sensors sensors_data
                if zed.get_sensors_data(sensors_data,
                                        sl.TIME_REFERENCE.IMAGE):  # time_reference.image for synchorinzed timestamps
                    # Check if the data has been updated since the last time
                    # IMU is the sensor with the highest rate
                    if ts_handler.is_new(sensors_data.get_imu_data()):
                        # Access the magnetometer data
                        magnetometer_data = sensors_data.get_magnetometer_data()
                        # Get the raw magnetic heading  # Apply low-pass filter
                        # magnetic_heading = magnetic_filter.update(sensors_data.get_magnetometer_data().magnetic_heading)
                        magnetic_heading = sensors_data.get_magnetometer_data().magnetic_heading

                        # Access the magnetic heading and state
                        magnetic_heading_info = (
                            f"Magnetic Heading: {magnetic_heading:.0f} "
                            f"({magnetometer_data.magnetic_heading_state}) "
                            f"[{magnetometer_data.magnetic_heading_accuracy:.1f}]"
                        )
                        render_text(frame, magnetic_heading_info, (frame.shape[1] - 1300, 30))

                # mevcut koordinatları al
                translation = pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
                current_x = translation[0]
                current_y = -(translation[1])

                draw_map_with_heading(current_x, current_y, magnetic_heading)

                # Her tespit kutusunun sağ üst köşesine derinlik değerini yazdırmak için:
                for box in coordinates:
                    x1, y1, x2, y2 = map(int, box)  # tamsayıya çeviriyoruz
                    # Sağ üst köşe koordinatları: (x2, y1)
                    depth_val = depth.get_value((x2 + x1) / 2, (y1 + y2) / 2)[
                        1]  # Eğer depth değeri geçerliyse (NaN değilse) yazdır
                    if not np.isnan(depth_val):
                        text = f"{depth_val:.2f} m"
                        # Yazıyı kutunun sağ üst köşesine ekleyelim; konum ayarını isteğinize göre değiştirebilirsiniz
                        cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)

                red_detected = False
                green_detected = False
                yellow_detected = False
                blue_detected = False
                black_detected = False
                cross_detected = False
                triangle_detected = False

                red_positions = []
                green_positions = []
                yellow_positions = []
                blue_positions = []
                black_positions = []
                cross_positions = []
                triangle_positions = []

                #while not (şekiller detected and şekil tespitinden 7 saniye geçmişse)
                for i, class_id in enumerate(class_ids):
                    if class_id == 4:  # Kırmızı
                        red_detected = True
                        red_positions.append(coordinates[i])
                    elif class_id == 6:  # Sarı
                        yellow_detected = True
                        yellow_positions.append(coordinates[i])
                    elif class_id == 3:  # Yeşil
                        green_detected = True
                        green_positions.append(coordinates[i])
                    elif class_id == 1:  # Mavi
                        blue_detected = True
                        blue_positions.append(coordinates[i])
                    elif class_id == 0:  # Siyah
                        black_detected = True
                        black_positions.append(coordinates[i])
                    elif class_id == 2:  # siyah artı
                        cross_detected = True
                        cross_positions.append(coordinates[i])
                    elif class_id == 5:  # siyah üçgen
                        triangle_detected = True
                        triangle_positions.append(coordinates[i])

                # Tuş kontrolü: 'm' tuşu ile modlar arasında geçiş yapılır.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m'):
                    manual_mode = not manual_mode
                    if manual_mode is True:
                        print("Manuel mod aktif. Otomatik sürüş durdu.")
                    else:
                        print("Otomatik mod aktif. Manuel kontrol devre dışı.")
                    # Küçük bir gecikme, tuşun sürekli algılanmasını önlemek için
                    time.sleep(0.2)

                # Manuel mod aktifse, WASD tuşlarıyla kontrol yapılır.
                if manual_mode:
                    cv2.putText(frame, "MANUEL MOD", (50, 50), FONT, 1, (0, 255, 255), 2)  # todo: ekran orta noktasına al
                    if key == ord('w'):
                        # İleri hareket: her iki motor ileri
                        print("manual")
                        controller.set_servo(5, 1600)
                        controller.set_servo(6, 1600)
                    elif key == ord('s'):
                        # Geri hareket: her iki motor geri
                        print("manual")
                        controller.set_servo(5, 1300)
                        controller.set_servo(6, 1300)
                    elif key == ord('a'):
                        # Sola dönüş: sol motor yavaş, sağ motor hızlı
                        print("manual")
                        controller.set_servo(5, 1420)
                        controller.set_servo(6, 1580)
                    elif key == ord('d'):
                        # Sağa dönüş: sol motor hızlı, sağ motor yavaş
                        print("manual")
                        controller.set_servo(5, 1580)
                        controller.set_servo(6, 1420)
                    elif key == ord('l'):
                        # Tuşlara basılmadığında motorlar nötr konumda kalır
                        controller.set_servo(5, 1500)
                        controller.set_servo(6, 1500)
                else:

                    # --- AUTONOMOUS MISSIONS (ALL MUST BE PLACED IN THIS ELSE CONDITION) ---

                    # Check if the special mission trigger condition is met
                    if special_mission_triggered():
                        # Save the current mission state if we're not already in the special mission
                        if mission_state != MISSION_SPECIAL:
                            prev_mission_state = mission_state
                            mission_state = MISSION_SPECIAL

                    if mission_state == MISSION_AVOID_BUOYS:
                        avoid_buoys(frame, depth, center_x, center_y,
                           green_detected, red_detected, yellow_detected,
                           blue_detected, black_detected,
                           green_positions, red_positions, yellow_positions)
                        cv2.putText(frame, "Avoiding buoys", (50, 350), FONT, 1, (255, 255, 0), 2)
                        # Transition condition example: if buoys have been successfully passed
                        # if buoys_passed_condition():
                        mission_state = MISSION_DOCKING

                    elif mission_state == MISSION_DOCKING:
                        done = docking(frame, depth, current_x, current_y, magnetic_heading, center_x,
                                                      center_y, map_image, detections)
                        cv2.putText(frame, "Docking", (50, 350), FONT, 1, (255, 255, 0), 2)
                        if done:
                            mission_state = MISSION_SPEED_CHALLENGE

                    elif mission_state == MISSION_SPEED_CHALLENGE:
                        # Placeholder for your second intermediate mission.
                        done = speed_challenge(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections)
                        cv2.putText(frame, "Speed challenge", (50, 350), FONT, 1, (255, 255, 0), 2)
                        if done:
                            mission_state = MISSION_RETURN_HOME

                    elif mission_state == MISSION_RETURN_HOME:
                        navigate_to_start(frame, current_x, current_y, magnetic_heading, start_x, start_y)
                        cv2.putText(frame, "Going back to home", (50, 350), FONT, 1, (255, 255, 0), 2)

                    elif mission_state == MISSION_SPECIAL:
                        done = special_mission(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections)
                        cv2.putText(frame, "Delivery Mission Active", (50, 400), FONT, 1, (255, 0, 255), 2)
                        if done:
                            # When done, revert back:
                            mission_state = prev_mission_state

                cv2.putText(frame, f"FPS: {int(zed.get_current_fps())}", (10, 30), FONT, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"{str(zed.get_spatial_mapping_state())}", (10, 60), FONT, 0.5, (20, 220, 20), 1)
                cv2.putText(frame, f"POSITIONAL_TRACKING_STATE.{str(zed.get_position(pose, sl.REFERENCE_FRAME.WORLD))}",
                            (10, 90), FONT, 0.5, (20, 220, 20), 1)
                cv2.putText(frame, f"Coordinates X,Y: {current_x:.1f} {current_y:.1f}", (10, 120), FONT, 0.75,
                            (0, 150, 240), 1, )

                # Görüntüyü göster
                frame_resized = cv2.resize(frame, (960, 540))  # Resize the frame to desired dimensions960, 540
                cv2.imshow("ZED Camera", frame_resized)

                if key % 256 == 27:
                    print("Esc tuşuna basıldı.. Kapatılıyor..")
                    controller.stop_motors()
                    break

        # Kaynakları serbest bırak ve kamerayı kapat
        cv2.destroyAllWindows()
        zed.close()
    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        # Kaynakları serbest bırak
        if 'zed' in locals():
            zed.close()
        print("Program sonlandırıldı.")


if __name__ == "__main__":
    main()
