import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from trackers.multi_tracker_zoo import create_tracker

class ObjectTracker:
    def __init__(self, source_video_path, save_path, model_path, video_write=False):
        self.source_video_path = source_video_path
        self.save_path = save_path
        self.video_write = video_write
        self.show_video_when_writing = show_video_when_writing
        self.video_saving_path = save_path if video_write else None

        self._create_default_https_context = _create_unverified_context
        self.model = torch.hub.load('ultralytics/yolov5', "custom", model_path, force_reload=False, device=0)
        self.model.to("cuda")
        self.names = self.model.names
        self.model.classes = [0]

        self.video_cap = cv2.VideoCapture(source_video_path)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.width, self.height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.desired_fps = 30

        if video_write:
            self.result = cv2.VideoWriter(self.video_saving_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

    def run(self):
        start_time = time()

        polygon_points = np.array([[283, 393], [484, 267], [551, 375], [305, 524]])
        polygon_points2 = np.array([[564, 391], [307, 551], [342, 754], [663, 547]])
        in_polyline = np.array([[564, 391], [307, 551]])
        out_polyline = np.array([[305, 524], [551, 375]])

        passing_dict = {}
        outcounter = 0
        incounter = 0

        tracker_list = []
        tracker = create_tracker('bytetrack', tracker_config_path, tracker_weights_path, device=torch.device("cuda"), half=False)
        tracker_list.append(tracker)

        count = 0
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()

            if not ret:
                break

            count += 1
            if count % 1 != 0:
                continue

            results = self.model(frame)
            det = results.xyxy[0]

            cv2.putText(frame, f"{incounter} IN ", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 229, 204), 3)
            cv2.putText(frame, f"{outcounter} OUT ", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 229, 204), 3)
            cv2.polylines(frame, np.int32([polygon_points]), True, (255, 0, 0), 3)
            cv2.polylines(frame, np.int32([polygon_points2]), True, (255, 0, 0), 3)
            cv2.polylines(frame, np.int32([in_polyline]), True, (255, 155, 255), 3)
            cv2.polylines(frame, np.int32([out_polyline]), True, (255, 155, 255), 3)

            if det is not None and len(det):
                outputs = tracker_list[0].update(det.cpu(), frame)
                
                for j, (output) in enumerate(outputs):
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2, c, id = int(x1), int(y1), int(x2), int(y2), int(cls), int(id)

                    center_x, center_y = int((x1 + x2) / 2), int(y2 - 10)

                    area_check_1 = cv2.pointPolygonTest(np.int32([polygon_points]), ((center_x, center_y)), False)
                    area_check_2 = cv2.pointPolygonTest(np.int32([polygon_points2]), ((center_x, center_y)), False)

                    in_color = (50, 255, 50)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), in_color, 2)
                    cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                    cv2.putText(frame, f"{self.names[int(c)]}{str(id)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)

                    if id not in passing_dict:
                        passing_dict[id] = "new"

                    if area_check_1 == 1:
                        if passing_dict[id] == "new":
                            passing_dict.update({id: "inx"})
                        if passing_dict[id] == "iny":
                            passing_dict.update({id: "inx"})
                            outcounter += 1

                    if area_check_2 == 1:
                        if passing_dict[id] == "new":
                            passing_dict.update({id: "iny"})
                        if passing_dict[id] == "inx":
                            passing_dict.update({id: "iny"})
                            incounter += 1

            if self.show_video_when_writing:
                cv2.imshow("ROI", frame)

            if self.video_write:
                print(f"frame {count} writing")
                self.result.write(frame)
            if cv2.waitKey(20) == ord('q'):
                break

        self.video_cap.release()
        if self.video_write:
            self.result.release()
        cv2.destroyAllWindows()
        print("process done")
        print("Execution time:", time() - start_time, "seconds")

if __name__ == "__main__":
    source_video_path = r"C:\Users\berk_\Downloads\Graduation-Thesis-main\videos\test.mp4"
    save_path = r"C:\Users\berk_\Downloads\Graduation-Thesis-main\outputs\output.mp4"
    model_path = r'C:\Users\berk_\Downloads\Graduation-Thesis-main\weights\tezv2.pt'
    tracker_config_path = r"C:\Users\berk_\Downloads\Graduation-Thesis-main\app\trackers\bytetrack\configs\bytetrack.yaml"
    tracker_weights_path = "weights/osnet_x0_25_msmt17.pt"
    video_write = True  # Set to True if you want to save the output video
    show_video_when_writing = True #Set to False if you dont want to show video when writing

    tracker = ObjectTracker(source_video_path, save_path, model_path, video_write)
    tracker.run()
