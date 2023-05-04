import cv2
from predict import predict_handshape

# how to add text: https://www.geeksforgeeks.org/python-opencv-write-text-on-video/
# how to take video: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

cam_port = 0
cam = cv2.VideoCapture(cam_port)
width  = cam.get(3)
height = cam.get(4)

while (True):
    # capture the frames in the video, flip so looks like the same hand
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # print(type(frame))
    print(frame.shape)

    # add box for where to place hand 
    end_x = int(width * 0.9)
    start_x = end_x - 400
    start_y = int(height * 0.25)
    end_y = start_y + 400
    cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), (255,0,0), 2)

    # crop to the box frame, resize
    box_img = frame[start_y:end_y, start_x:end_x]
    img = cv2.resize(box_img, (200,200), 0.5, 0.5)

    # feed that image into the predict method of the model
    pred_letter = predict_handshape(img)
    print(pred_letter)

    # add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 
                pred_letter, 
                (int(width*0.15), int(width*0.15)), 
                font, 4, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

    # display frame
    cv2.imshow('frame', frame)

    # hit q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# after loop release cam object
cam.release()
# destroy all the windows
cv2.destroyAllWindows()