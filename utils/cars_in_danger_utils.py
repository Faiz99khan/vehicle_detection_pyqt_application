import cv2
import numpy as np

def finding_cars_in_danger(num_detections, score_thresh, scores, boxes, classes,safety_distance):
    dist_threshold={'h_dist':0,'v_dist':0,'d_dist':0,'c_dist':0}
    car_class=2 # 2 in yolo coco 2014, 3 coco 2107
    selected_boxes=[]
    car_avg_width=1.9  ## in meters from website
    cars_in_danger=[]
    for i in range(num_detections):
        if classes[i]==car_class and  score_thresh <= scores[i]:
            selected_boxes.append(i)

    for i,ref_box_index in enumerate(selected_boxes):
        ref_box=boxes[ref_box_index]
        ref_box= (ref_box[0],ref_box[1], ref_box[2], ref_box[3])

        ref_box=list(map(int,ref_box))
        (xrmin,yrmin, xrmax, yrmax)= ref_box
        #    print(ref_box)

        car_pixel_width=xrmax-xrmin   # approx width in pixels
        #  print(car_pixel_width)
        dist_threshold['h_dist']=.1*safety_distance*(car_pixel_width/car_avg_width)     ## threshold in pixels in horizontal direction
        dist_threshold['f_dist']=.1*(safety_distance-.1)*(car_pixel_width/car_avg_width)     ## threshold in pixels in forward direction
        dist_threshold['b_dist']=.1*safety_distance*(car_pixel_width/car_avg_width)     ## threshold in pixels in backward direction
        dist_threshold['d_dist']=.1*safety_distance*(car_pixel_width/car_avg_width)     ## threshold in pixels in diagonal direction
        dist_threshold['c_dist']=.1*safety_distance*(car_avg_width + car_pixel_width/car_avg_width)     ## threshold in pixels from centers

        for box_index in selected_boxes[i+1:]:
            box=boxes[box_index]


            #box = (box[1], box[0],box[3],box[2])
            box = (box[0], box[1],box[2],box[3])

            box=list(map(int,box))
            (xmin,ymin, xmax, ymax)=box
            #    print(box)
            try:
                ### calculation with backward car
                if ymin>yrmax and (ymin-yrmax)<=dist_threshold['b_dist']:
                    if xrmin<=xmin<=xrmax or xrmin<=xmax<=xrmax:
                        closest_distance=ymin-yrmax
                        cars_in_danger.append([ref_box,box])
                    else:
                        if xrmin>xmax:
                            closest_distance=np.sqrt((xrmin-xmax)**2+(ymin-yrmax)**2)
                        else:
                            closest_distance=np.sqrt((xmin-xrmax)**2+(ymin-yrmax)**2)

                        if closest_distance<dist_threshold['d_dist']:
                            cars_in_danger.append([ref_box,box])


                ### calculation with forward car
                elif ymax<yrmin and (yrmin-ymax)<=dist_threshold['f_dist']:
                    if xrmin<=xmin<=xrmax or xrmin<=xmax<=xrmax:
                        closest_distance=yrmin-ymax
                        cars_in_danger.append([ref_box,box])
                    else:
                        if xrmin>xmax:
                            closest_distance=np.sqrt((xrmin-xmax)**2+(yrmin-ymax)**2)
                        else:
                            closest_distance=np.sqrt((xmin-xrmax)**2+(yrmin-ymax)**2)

                        if closest_distance<dist_threshold['d_dist']:
                            cars_in_danger.append([ref_box,box])

                ### calculation with left car
                elif xmax<xrmin and (xrmin-xmax)<=dist_threshold['h_dist']:
                    if yrmin<=ymin<=yrmax or yrmin<=ymax<=yrmax:
                        closest_distance=xrmin-xmax
                        cars_in_danger.append([ref_box,box])

                ### calculation with right car
                elif xmin>xrmax and (xmin-xrmax)<=dist_threshold['h_dist']:
                    if yrmin<=ymin<=yrmax or yrmin<=ymax<=yrmax:
                        closest_distance=xmin-xrmax
                        cars_in_danger.append([ref_box,box])
                else:
                    # calculation with overlapping car
                    X_min=xrmin if xrmin>xmin else xmin
                    Y_min=yrmin if yrmin>ymin else ymin
                    X_max=xrmax if xrmax<xmax else xmax
                    Y_max=yrmax if yrmin<ymin else ymax
                    if (X_max-X_min)>0 and (Y_max-Y_min)>0:
                        cars_in_danger.append([ref_box,box])

            except Exception as e:
                print('exception',e)
        # print(cars_in_danger)

    return cars_in_danger

def locate_cars_in_danger(cars_in_danger,frame):
    try:
        for box1,box2 in cars_in_danger:
            box1_pt1=(box1[0],box1[1])
            box1_pt2=(box1[2],box1[3])

            box2_pt1=(box2[0],box2[1])
            box2_pt2=(box2[2],box2[3])

            center1=((box1[2]+box1[0])//2,(box1[3]+box1[1])//2)
            center2=((box2[2]+box2[0])//2,(box2[3]+box2[1])//2)

            cv2.rectangle(frame,box1_pt1,box1_pt2,(0,0,255),2)
            cv2.rectangle(frame,box2_pt1,box2_pt2,(0,0,255),2)
            cv2.line(frame,center1,center2,(0,0,255),2)

    except Exception as e:
        print('Exception',e)
