import cv2
import os


def capture_frames(video_path, output_folder, intervals, photos_per_second=7):
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    # Video özelliklerini al
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Kullanıcıdan hangi kesitlerde fotoğraf çekileceğini al
    for start_time, end_time in intervals:
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = start_frame
        while frame_number <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Her saniyede belirli bir sayıda fotoğraf almak için uygun kareyi kontrol et
            if (frame_number - start_frame) % (fps // photos_per_second) == 0:
                frame_filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
                cv2.imwrite(frame_filename, frame)

            frame_number += 1

    cap.release()
    print("Fotoğraf çekme işlemi tamamlandı.")


if __name__ == "__main__":
    video_path = 'SEKILVideo.mp4'  # Videonuzun yolu
    print("video path alındı")
    output_folder = 'balonFoto'  # Fotoğrafların kaydedileceği klasör
    print("output path alındı")
    intervals = [( 0, 43 )]  # Fotoğraf çekmek istediğiniz zaman kesitleri (saniye cinsinden)


    capture_frames(video_path, output_folder, intervals, photos_per_second=3)
    print("fotolar sikildi")
