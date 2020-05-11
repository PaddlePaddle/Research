import json
import os
import cv2
from random import randrange

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]
videoInfo = {"cam_1": {"frame_num": 3000, "movement_num": 4},
             "cam_1_dawn": {"frame_num": 3000, "movement_num": 4},
             "cam_1_rain": {"frame_num": 2961, "movement_num": 4},
             "cam_2": {"frame_num": 18000, "movement_num": 4},
             "cam_2_rain": {"frame_num": 3000, "movement_num": 4},
             "cam_3": {"frame_num": 18000, "movement_num": 4},
             "cam_3_rain": {"frame_num": 3000, "movement_num": 4},
             "cam_4": {"frame_num": 27000, "movement_num": 12},
             "cam_4_dawn": {"frame_num": 4500, "movement_num": 12},
             "cam_4_rain": {"frame_num": 3000, "movement_num": 12},
             "cam_5": {"frame_num": 18000, "movement_num": 12},
             "cam_5_dawn": {"frame_num": 3000, "movement_num": 12},
             "cam_5_rain": {"frame_num": 3000, "movement_num": 12},
             "cam_6": {"frame_num": 18000, "movement_num": 12},
             "cam_6_snow": {"frame_num": 3000, "movement_num": 12},
             "cam_7": {"frame_num": 14400, "movement_num": 12},
             "cam_7_dawn": {"frame_num": 2400, "movement_num": 12},
             "cam_7_rain": {"frame_num": 3000, "movement_num": 12},
             "cam_8": {"frame_num": 3000, "movement_num": 6},
             "cam_9": {"frame_num": 3000, "movement_num": 12},
             "cam_10": {"frame_num": 2111, "movement_num": 3},
             "cam_11": {"frame_num": 2111, "movement_num": 3},
             "cam_12": {"frame_num": 1997, "movement_num": 3},
             "cam_13": {"frame_num": 1966, "movement_num": 3},
             "cam_14": {"frame_num": 3000, "movement_num": 2},
             "cam_15": {"frame_num": 3000, "movement_num": 2},
             "cam_16": {"frame_num": 3000, "movement_num": 2},
             "cam_17": {"frame_num": 3000, "movement_num": 2},
             "cam_18": {"frame_num": 3000, "movement_num": 2},
             "cam_19": {"frame_num": 3000, "movement_num": 2},
             "cam_20": {"frame_num": 3000, "movement_num": 2}}

if __name__ == "__main__":
    # segment number
    n = 10 

    jsonRoot = "../"
    imgRoot = "../../screen_shot_with_roi_and_movement/"
    visRoot = "../vis/"
    for vId, vName in enumerate(videoInfo.keys()):
        camName = "cam_" + vName.split("_")[1]
        img = cv2.imread(imgRoot + camName + ".jpg")
        with open(jsonRoot + camName + ".json") as fp:
            camJson = json.load(fp)
            for mName, moveInfo in camJson.items():
                colVal = tuple((randrange(255), randrange(255), randrange(255), 0.3))
                mId = int(mName.split("_")[-1])
                x1, y1 = moveInfo["src"]["point_1"]
                x2, y2 = moveInfo["src"]["point_2"]
                #import pdb;pdb.set_trace()
                cv2.line(img, (x1,y1), (x2,y2), colVal, 5)
                cv2.putText(img, '%ds'%(mId), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colVal, 2)

                x3, y3 = moveInfo["dst"]["point_1"]
                x4, y4 = moveInfo["dst"]["point_2"]
                cv2.line(img, (x3,y3), (x4,y4), colVal, 5)
                cv2.putText(img, '%de'%(mId), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colVal, 2)

                # draw tracklets
                tracklet_key = "tracklets"
                if tracklet_key in moveInfo.keys():
                    for m_tracklet in moveInfo["tracklets"]:
                        print(mId)
                        #import pdb;pdb.set_trace()
                        p0 = tuple(m_tracklet[0])
                        cv2.putText(img, '%ds'%(mId), (int(p0[0]), int(p0[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colVal, 2)
                        for i in range(1, len(m_tracklet)):
                            p1 = tuple(m_tracklet[i])
                            # cv2.line(img, p0, p1, colVal, 5)
                            cv2.arrowedLine(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), colVal, 2)
                            p0 = p1

        cv2.imwrite(visRoot + camName + ".jpg", img)
        print(visRoot + camName + ".jpg")
