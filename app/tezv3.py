#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from trackers.multi_tracker_zoo import create_tracker

start_time = time()

_create_default_https_context = _create_unverified_context

source_video_path = r"C:\Users\berk_\Downloads\Graduation-Thesis-main\videos\test.mp4"
save_path = r"C:\Users\berk_\Downloads\Graduation-Thesis-main\outputs"
video_saving_path = save_path[:len(save_path)-4:]+"_output.mp4"


model = torch.hub.load('ultralytics/yolov5', 
                       "custom", 
                       r'C:\Users\berk_\Downloads\Graduation-Thesis-main\weights\tezv2.pt', 
                       force_reload=False, 
                       device= 0)

model.to("cuda")
names = model.names #to get class names 
model.classes = [0]
#model.classes = [None]
video_write = False

#to use with default yolo model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_cap=cv2.VideoCapture(source_video_path)

fps = video_cap.get(cv2.CAP_PROP_FPS)
width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

desired_fps = 30

if video_write:
    result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))

########################POLYGON############################

polygon_points= np.array(   [[283, 393], [484, 267], [551, 375], [305, 524]])
polygon_points2 = np.array( [[564, 391], [307, 551], [342, 754], [663, 547]])

in_polyline = np.array([[564, 391],[307, 551]])
out_polyline = np.array([[305, 524],[551, 375]])

passing_dict= {}
outcounter = 0
incounter = 0


########################TRACKER###########################
tracker_list = []
tracker = create_tracker('bytetrack', 
                         r"C:\Users\berk_\Downloads\Graduation-Thesis-main\app\trackers\bytetrack\configs\bytetrack.yaml", 
                         "weights/osnet_x0_25_msmt17.pt", 
                         device=torch.device("cuda"), 
                         half =False)
tracker_list.append(tracker, )



########################MAIN LOOP###########################

count=0
while video_cap.isOpened():

    ret,frame=video_cap.read()
    #det_frame = frame[1070:330, 750:150]

    if not ret:
        break
    #ADJUST FPS
    count +=1
    if count % 1 != 0:
        continue

    results = model(frame)
    det = results.xyxy[0]
    cv2.putText(frame, f"{incounter} IN ", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,229,204), 3)
    cv2.putText(frame, f"{outcounter} OUT ", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,229,204), 3)
    cv2.polylines(frame, np.int32([polygon_points]), True, (255,0,0),3)
    cv2.polylines(frame, np.int32([polygon_points2]), True, (255, 0, 0), 3)

    cv2.polylines(frame, np.int32([in_polyline]), True, (255,155,255),3)
    cv2.polylines(frame, np.int32([out_polyline]), True, (255, 155, 255), 3)
    if det is not None and len(det): #work if there is detections
        outputs = tracker_list[0].update(det.cpu(), frame)
        
        for j, (output) in enumerate(outputs):
            bbox = output[0:4]
            id = output[4]
            cls = output[5]
            conf = output[6]
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2, c, id= int(x1),int(y1),int(x2), int(y2), int(cls), int(id)
            ###############################ALGORITHM WORK HERE###################################
            center_x, center_y= int((x1+x2)/2), int(y2-10)

            area_check_1 = cv2.pointPolygonTest(np.int32([polygon_points]),((center_x,center_y)), False)
            area_check_2 = cv2.pointPolygonTest(np.int32([polygon_points2]), ((center_x, center_y)), False)


            if area_check_1 == 1:
                in_color = (50,255,50)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)

            else:
                in_color = (50,255,50)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)

            if area_check_2 == 1:
                in_color = (50,255,50)
                cv2.rectangle(frame, (x1, y1), (x2, y2), in_color, 2)
                cv2.circle(frame, (center_x, center_y), radius=3,
                           color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)

            else:
                in_color = (50,255,50)
                cv2.rectangle(frame, (x1, y1), (x2, y2), in_color, 2)
                cv2.circle(frame, (center_x, center_y), radius=3,
                           color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)


            if id not in passing_dict:
                passing_dict[id] = "new"
            print(passing_dict)

            if area_check_1 == 1:
                if passing_dict[id] == "new":
                    passing_dict.update({id:"inx"})
                    #object_counter += 1
                if passing_dict[id] == "iny":
                    passing_dict.update({id: "inx"})
                    outcounter += 1
            
            if area_check_2 == 1:
                if passing_dict[id] == "new":
                    passing_dict.update({id: "iny"})
                    #object_counter += 1
                if passing_dict[id] == "inx":
                    passing_dict.update({id: "iny"})
                    incounter += 1
                    
        print(passing_dict)

    if not video_write:
        cv2.imshow("ROI",frame)

    if video_write:
        print(f"frame {count} writing")
        result.write(frame)
    if cv2.waitKey(20) == ord('q'):
        break


video_cap.release()
if video_write:
    result.release()
cv2.destroyAllWindows()
print("process done")
print("Execution time:", time() - start_time, "seconds")