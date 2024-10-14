#!/usr/bin/env python3
# -- coding: utf-8 --
import rospy
import torch
import cv2
import numpy as np
import psutil
import subprocess

from enum import IntEnum
from typing import Tuple
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from std_msgs.msg import Float64

PT_PATH = r"C:\Users\H_\Desktop\Sungmin_Github\VisionTeamStudy\홍성민\Lane_Pt_Files\only_traffic_lane_2.pt"
VIDEO_PATH = r"C:\Users\H_\Desktop\Sungmin_Github\VisionTeamStudy\홍성민\KakaoTalk_20240803_180550455.mp4"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = YOLO(PT_PATH).to(DEVICE)

steer_angle_pub = rospy.Publisher('/steer_angle', Float64, queue_size=1)
left_heading_distance_pub = rospy.Publisher('/left_heading_distance', Float64, queue_size=1)
right_heading_distance_pub = rospy.Publisher('/right_heading_distance', Float64, queue_size=1)


class lane_shape_info(IntEnum):
    '''문자열을 숫자로 비교하기 위함입니다.
    이는 실행 속도를 향상시킵니다.
    '''
    Curve = 1
    Straight = 2

class steer_info(IntEnum):
    '''위와 똑같은 이유로 썼습니다.
    '''
    Straight = 1
    Left = 2
    Right = 3
    Safe = 4

def hardware_check():
    '''코드를 실행함으로써 CPU, Memory, GPU 사용량을 알아보기 위함입니다.
    Args
        None
    
    Returns
        None
    '''
    # CPU
    cpu_percent = psutil.cpu_percent()

    # Memory
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  
    available_memory = memory_info.available / (1024 ** 3) 
    used_memory = total_memory - available_memory

    # GPU
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], 
                            stdout=subprocess.PIPE)
    gpu_usage = result.stdout.decode('utf-8').strip()
    
    print('='*30)
    print(f'사용 중인 CPU: {cpu_percent}%')
    print(f"전체 메모리: {total_memory:.2f}GB")
    print(f"사용 가능한 메모리: {available_memory:.2f}GB")
    print(f"사용 중인 메모리: {used_memory:.2f}GB")
    print(f'사용 중인 GPU : {gpu_usage}%')
    print('='*30)


def calibration()-> Tuple[np.ndarray, np.ndarray]:
    '''intrinsic : DarkProgrammer의 intrinsic을 얻는 tool을 통해 가지고 있는 camera로 checkboard를 찍어 얻게 된다.
    extrinsic paramter : cv2.solvePnP(object points, image points, cameraIntrinsic, distCoeffs) -> ret, rvec(rotation vector), tvec(translate vector)를 반환한다.
    rvec은 cv2.Rodrigues로 변환해줘야합니다.
    
    Args
        None

    Returns
        intrinsic, extrinsic
    '''
    f = 345.727618 # fx, fy
    cx, cy = 320.000000, 240.000000 

    cameraMatrix = np.array([[f, 0, cx],
                             [0, f, cy],
                             [0, 0, 1]])    
    
    rotationMatrix = np.array([[0.07517367, -0.99703638, 0.01635175],
                               [-0.07233739, -0.02180751, -0.99714178],
                               [0.99454322, 0.07377597, -0.07376236]],
                               dtype=np.float32)
    translateMatrix = np.array([[-0.3260739],
                                [1.48923499],
                                [-1.41693029]], dtype=np.float32)
    
    extrinsicMatrix = np.hstack((rotationMatrix, translateMatrix))
    extrinsicMatrix = np.vstack((extrinsicMatrix, [0, 0, 0, 1]))  # 4x4 매트릭스로 변환

    return cameraMatrix, extrinsicMatrix

def precompute_mapping(cameraMatrix: np.ndarray, extrinsicMatrix: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    '''BEV를 만들어, 1개의 pixel당 몇 m를 알아낼 수 있는 방법이다. (URL : https://gaussian37.github.io/vision-concept-ipm/)

    x : height
    y : width

    Args
        None
    
    Returns
        BEV Frame
    '''
    world_height_max, world_height_min = 11, 0.5 # 단위 : m
    world_width_max, world_width_min = 4, -4 

    world_height_interval = 0.05
    world_width_interval = 0.025

    output_height = int(np.ceil((world_height_max - world_height_min) / world_height_interval))
    output_width = int(np.ceil((world_width_max - world_width_min) / world_width_interval))

    world_height_coords = np.arange(world_height_max, world_height_min, -world_height_interval)
    world_width_coords = np.arange(world_width_max, world_width_min, -world_width_interval)

    output_height = len(world_height_coords)
    output_width = len(world_width_coords)

    map_height = np.zeros((output_height, output_width)).astype(np.float32)
    map_width = np.zeros((output_height, output_width)).astype(np.float32)


    for i, world_height in enumerate(world_height_coords):
        for j, world_width in enumerate(world_width_coords):
            world_coord = [world_height, world_width, 0, 1]
            camera_coord = extrinsicMatrix[:3, :] @ world_coord
            uv_coord = cameraMatrix @ camera_coord
            uv_coord /= uv_coord[2]

            map_height[i][j] = uv_coord[0]
            map_width[i][j] = uv_coord[1]

    return map_height, map_width

def define_world_coordinate(frame: np.ndarray, map_height: np.ndarray, map_width: np.ndarray)-> np.ndarray:
    '''BEV를 만들어, 1개의 pixel당 몇 m를 알아낼 수 있는 방법이다. (URL : https://gaussian37.github.io/vision-concept-ipm/)

    x : height
    y : width

    Args
        frame : 원본 frame
    
    Returns
        BEV Frame
    '''
    lut_frame = cv2.remap(frame, map_height, map_width, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return lut_frame

def extract_mask_regression(mask: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    '''
    Args
        masks : 차선 segmentation된 정보들
    
    Returns
        x, y_fit : width, height로 회귀 정보들
    '''
    x = mask[:, 0].reshape(-1, 1)
    y = mask[:, 1]

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)

    regress_model = LinearRegression()
    regress_model.fit(x_poly, y)

    x_min, x_max = x.min(), x.max()
    x_new = np.linspace(x_min, x_max, 40).reshape(-1, 1)
    x_poly_new = poly.transform(x_new)  # 수정된 부분: poly.fit_transform -> poly.transform

    y_fit_new = regress_model.predict(x_poly_new)

    return x_new, y_fit_new

def calculate_radiusCurvature(coeffs: np.ndarray, specific_coord: np.int64)-> np.float64:
    '''곡률을 구하고 곡률 반경을 구합니다.
    "https://www.youtube.com/watch?v=2oQnljpQm4Y"를 참조하였습니다.

    Args
        coeffs : Ax^2 + Bx + C
        specifit_coord : 차선에서의 특정 1개의 좌표를 의미

    Return
        curvature : 차선의 곡률 반경
    '''
    A = coeffs[0]
    B = coeffs[1]

    first_order = 2 * A * specific_coord + B
    second_order = 2 * A

    curvature = np.abs(second_order) / (1 + first_order ** 2) ** (3 / 2)
    radius_curvature = 1 / curvature

    return radius_curvature

def ipm_lane_info(bev_frame: np.ndarray)-> IntEnum:
    '''BEV에서의 차선의 곡률(기울기)값을 구합니다.
    이는, 차선을 위에서 아래로 바라봄으로써 차선의 왜곡현상을 제거하여 효과적인 차선 분석이 가능합니다.
    
    left : (255, 0, 0)
    right : (0, 0, 255)

    Args
        bev_frame : bird eye view frame
        
    Return
        2개가 같으면 해당 하는 1개의 값을 보내주고, 그렇지 않으면 left의 정보를 보내줍니다.
    '''
    left_lane = cv2.inRange(bev_frame, (255, 0, 0), (255, 0, 0))
    right_lane = cv2.inRange(bev_frame, (0, 0, 255), (0, 0, 255))

    left_lane_coord = np.argwhere(left_lane)
    right_lane_coord = np.argwhere(right_lane)

    left_lane_shape = None
    right_lane_shape = None

    if np.any(left_lane_coord):
        left_height = left_lane_coord[:, 0]
        left_width = left_lane_coord[:, 1]
        
        left_coeffs = np.polyfit(left_height, left_width, 2) # Ax^2 + Bx + C -> A, B, C
        left_radius_curvature = calculate_radiusCurvature(left_coeffs, np.max(left_height))
        if left_radius_curvature < 700:
            left_lane_shape = lane_shape_info.Curve
        else:
            left_lane_shape = lane_shape_info.Straight

    if np.any(right_lane_coord):
        right_height = right_lane_coord[:, 0]
        right_width = right_lane_coord[:, 1]

        right_coeffs = np.polyfit(right_height, right_width, 2)
        right_radius_curvature = calculate_radiusCurvature(right_coeffs, np.max(right_height))
        if right_radius_curvature < 700:
            right_lane_shape = lane_shape_info.Curve
        else:
            right_lane_shape = lane_shape_info.Straight

    if right_lane_shape == left_lane_shape:
        return right_lane_shape
    else:
        return left_lane_shape

def draw_lanes(lane_idx: int, frame: np.ndarray, x_fit: np.ndarray, y_fit: np.ndarray)-> np.ndarray:
    '''차선에 대한 정보를 frame에 그립니다.
       lane_idx는 0이면 왼쪽, 1이면 오른쪽을 의미합니다.

    Args
        frame : overlay의 정보로 frame의 복사본입니다.
        x_fit : x의 회귀 정보
        y_fit : y의 회귀 정보

    Return
        frame : overlay에 차선에 대한 정보를 그린 frame입니다.
    '''
    
    left_color, right_color = (255, 0, 0), (0, 0, 255)
    radius, thickness = 2, -1
    if lane_idx == 0:
        for (x, y) in zip(x_fit.flatten(), y_fit.flatten()):
            cv2.circle(frame, (int(x), int(y)), radius, left_color, thickness)
    elif lane_idx == 1:
        for (x, y) in zip(x_fit.flatten(), y_fit.flatten()):
            cv2.circle(frame, (int(x), int(y)), radius, right_color, thickness)

    return frame

def draw_center_lanes(frame: np.ndarray, concat_left_lane: np.ndarray, concat_right_lane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Args
        left_lane : 왼쪽 차선
        right_lane : 오른쪽 차선
    
    Return
        frame : 중앙 차선이 그려진 frame

    '''
    center_lane = []
    center_color = (0, 255, 0)
    radius, thickness = 2, -1

    sorted_concat_left_lane = sorted(concat_left_lane, key=lambda x: x[1], reverse=True)
    sorted_concat_right_lane = sorted(concat_right_lane, key=lambda x: x[1], reverse=True)
    
    for (lx, ly), (rx, ry) in zip(sorted_concat_left_lane, sorted_concat_right_lane):
        center_x = int((lx + rx) / 2)
        center_y = int((ly + ry) / 2)
        center = [center_x, center_y]
        center_lane.append(center)
        cv2.circle(frame, (center_x, center_y), radius, center_color, thickness)

    return frame, np.array(center_lane)

def adjust_vehicle_direction(frame: np.ndarray, center_lane: np.ndarray)-> Tuple[np.ndarray, IntEnum]:
    '''ERP-42가 바라보는 곳과 중앙선의 좌표 추종으로 steer를 꺾을 수 있게 만듭니다.
    center_lane은 [width, height] 정보를 받아옵니다.
    Arg
        frame: 왼, 오, 중앙선의 차선을 모두 그린 frame
        center_lane : 중앙 차선을 의미합니다.
    
    Return
        frame: 차량이 바라보고 있는 곳을 frame에 그립니다.
    '''
    vehicle_direction_height_max = int(frame.shape[0])
    vehicle_direction_height_min = int(frame.shape[0] // 2) + 10
    vehicle_direction_width = int(frame.shape[1] // 2)
    
    vehicle_top = (vehicle_direction_width, vehicle_direction_height_min)
    vehicle_bottom = (vehicle_direction_width, vehicle_direction_height_max)

    steer = steer_info.Straight
    if np.any(center_lane):
        center_lane_width_critertion = center_lane[:, 0][-1]
        if center_lane_width_critertion > vehicle_direction_width:
            steer = steer_info.Right
        elif center_lane_width_critertion < vehicle_direction_width:
            steer = steer_info.Left

    cv2.line(frame, vehicle_top, vehicle_bottom, (0, 255, 255), 2)

    return frame, steer

def angle_of_steer(center_lane: np.ndarray)-> np.float64:
    '''중앙선과 ERP-42가 바라보는 각도를 구해줍니다.
    이는 cos(θ) = (a * b) / (|a| |b|)로 각각의 방향벡터로 구할 수 있습니다.
    
    Args
        center_lane : 중앙선을 의미합니다.

    Return
        angle: 중앙선과 ERP-42가 바라보는 각도 값을 반환합니다.

    '''
    if center_lane is None or len(center_lane) <= 1:
        return 0
    
    erp_direction_unit_vector = np.array([0, -1])  # a

    center_lane_direction_vector = np.array([center_lane[-1][0] - center_lane[0][0], center_lane[-1][1] - center_lane[0][1]]) # b
    norm_center_lane = np.linalg.norm(center_lane_direction_vector)
    
    if norm_center_lane == 0:
        return 0

    center_lane_direction_unit_vector = center_lane_direction_vector / norm_center_lane  

    dot_product = np.dot(erp_direction_unit_vector, center_lane_direction_unit_vector)
    dot_product = np.clip(dot_product, -1.0, 1.0) # arccos은 -1 ~ 1의 치역을 가지고 있음.
    angle_rad = np.arccos(dot_product)

    angle_degree = np.rad2deg(angle_rad)
    angle_degree = np.clip(angle_degree, -20, 20)

    return angle_degree

def exception_extract_centerLane_of_bothLanes(frame: np.ndarray)-> Tuple[float, float]:
    '''왼/오른쪽의 차선을 잘 인지하여도, 중앙선과 헤딩의 방향벡터가 평행에 가깝다면 중앙선으로 경로를 추종하지 못하게 됩니다.
    따라서, heading과 왼쪽/오른쪽 차선과의 유클리디안 거리를 구하게 됩니다.
    
    standard_lane_area는 일반적인 도로 폭 값입니다.
    Args:
        bev_frame: ipm으로 만든 이미지입니다.
    
    Returns:
        additional_steer_info: 왼쪽(Left)으로 갈지, 오른쪽(Right)으로 갈지에 대한 정보입니다. 만약 조건문안에 들지 않는다면 None을 반환하는 것에 주의하세요.
    '''
    standard_lane_area = 325 # cm
    real_left_to_heading_distance, real_right_to_heading_distance = 0, 0
    
    left_lane = cv2.inRange(frame, (255, 0, 0), (255, 0, 0))
    right_lane = cv2.inRange(frame, (0, 0, 255), (0, 0, 255))
    heading_lane = cv2.inRange(frame, (0, 255, 255), (0, 255, 255))
    
    left_lane_index = np.argwhere(left_lane > 0) # 내림 차순 (height, width)
    right_lane_index = np.argwhere(right_lane > 0)
    heading_lane_index = np.argwhere(heading_lane > 0)
    
    if np.any(right_lane_index) and np.any(left_lane_index) and np.any(heading_lane_index):
        left_bottom = left_lane_index[np.argmax(left_lane_index[:, 0])]
        right_bottom = right_lane_index[np.argmax(right_lane_index[:, 0])]
        heading_bottom = heading_lane_index[np.argmax(heading_lane_index[:, 0])]
        
        right_to_left_pixel_distance = int(np.linalg.norm(right_bottom - left_bottom))
        left_to_heading_pixel_distance = int(np.linalg.norm(left_bottom - heading_bottom))
        right_to_heading_pixel_distance = int(np.linalg.norm(right_bottom - heading_bottom))
        
        real_lane_area = standard_lane_area / right_to_left_pixel_distance 
        real_left_to_heading_distance = left_to_heading_pixel_distance * real_lane_area
        real_right_to_heading_distance = right_to_heading_pixel_distance * real_lane_area
        
    return real_left_to_heading_distance, real_right_to_heading_distance        
        
def video_record(output_width: int, output_height: int, fps: float) -> cv2.VideoWriter:
    '''녹화를 위해 사용합니다.
    Args
        output_width : frame의 width
        output_height : frame의 height
        fps : cv2.imshow의 fps
    
    Return
        record_method : 동영상을 저장하는 class를 반환합니다.
    
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    record_method = cv2.VideoWriter(f'yolo_lane_detect.mp4', fourcc, fps, (output_width, output_height))
    return record_method

def detect_lines():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # == For Record result video == 
    # output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    
    # record = video_record(output_width, output_height, fps)
    
    cameraMatrix, extrinsicMatrix = calibration()
    map_height, map_width = precompute_mapping(cameraMatrix, extrinsicMatrix)
    
    prev_left_lane = np.array([])
    prev_right_lane = np.array([])
    
    both_detected_info = None

    prev_steer_angle = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        masks = []
        results = MODEL(frame)

        concat_left_lane = np.array([])
        concat_right_lane = np.array([])
        center_lane = np.array([])

        exception_concat_left_lane = np.empty((0, 2))
        exception_concat_right_lane = np.empty((0, 2))
        for result in results:
            if result.masks:
                masks = result.masks.xy
            
            detect_class_index = np.array(result.boxes.cls.tolist())
            detect_class_cnt = len(detect_class_index)
            
            if detect_class_cnt == 2:  # 2개 이하의 차선
                for mask in masks:
                    if np.any(mask):
                        x_fit, y_fit = extract_mask_regression(mask)
                        counts = sum(np.sum(x_fit < (frame.shape[1] // 2), axis=1))
                        expand_dim_y = y_fit.reshape(-1, 1)
                        if counts >= 10:
                            concat_left_lane = np.hstack((x_fit, expand_dim_y))
                            exception_concat_left_lane = np.vstack((exception_concat_left_lane, concat_left_lane))
                            if exception_concat_left_lane.size == 160:
                                right_x_fit, right_y_fit = exception_concat_left_lane[:40, 0], exception_concat_left_lane[:40, 1]
                                left_x_fit, left_y_fit = exception_concat_left_lane[40:, 0], exception_concat_left_lane[40:, 1]

                                concat_left_lane = np.array([])
                                concat_right_lane = np.array([])
                                
                                expand_dim_right_x, expand_dim_right_y = right_x_fit.reshape(-1, 1), right_y_fit.reshape(-1, 1)
                                expand_dim_left_x, expand_dim_left_y = left_x_fit.reshape(-1, 1), left_y_fit.reshape(-1, 1)
                                concat_left_lane = np.hstack((expand_dim_left_x, expand_dim_left_y))
                                concat_right_lane = np.hstack((expand_dim_right_x, expand_dim_right_y))

                                overlay = draw_lanes(0, overlay, left_x_fit, left_y_fit)
                                overlay = draw_lanes(1, overlay, right_x_fit, right_y_fit)
                            
                            else:  
                                overlay = draw_lanes(0, overlay, x_fit, y_fit)
                        else:
                            concat_right_lane = np.hstack((x_fit, expand_dim_y))
                            exception_concat_right_lane = np.vstack((exception_concat_right_lane, concat_right_lane))
                            if exception_concat_right_lane.size == 160:
                                left_x_fit, left_y_fit = exception_concat_right_lane[40:, 0], exception_concat_right_lane[40:, 1]
                                right_x_fit, right_y_fit = exception_concat_right_lane[:40, 0], exception_concat_right_lane[:40, 1]

                                concat_left_lane = np.array([])
                                concat_right_lane = np.array([])
                                
                                expand_dim_left_x, expand_dim_left_y = left_x_fit.reshape(-1, 1), left_y_fit.reshape(-1, 1)
                                expand_dim_right_x, expand_dim_right_y = right_x_fit.reshape(-1, 1), right_y_fit.reshape(-1, 1)
                                
                                concat_left_lane = np.hstack((expand_dim_left_x, expand_dim_left_y))
                                concat_right_lane = np.hstack((expand_dim_right_x, expand_dim_right_y))
                                
                                # 여기만 한정해서, index바꿔줌 left: 1, right: 0 (굳이 sorting을 사용해서 왼/오른쪽 나누지 않음 -> 연산량 신경씀)
                                overlay = draw_lanes(1, overlay, left_x_fit, left_y_fit)
                                overlay = draw_lanes(0, overlay, right_x_fit, right_y_fit)
                            
                            else:
                                overlay = draw_lanes(1, overlay, x_fit, y_fit)
                    
                    if np.any(concat_left_lane) and np.any(concat_right_lane):
                       overlay, center_lane = draw_center_lanes(overlay, concat_left_lane, concat_right_lane)
                       prev_left_lane, prev_right_lane = concat_left_lane, concat_right_lane
                       both_detected_info = "Current Lanes: Detected"
                       
        if detect_class_cnt != 2 and np.any(prev_left_lane) and np.any(prev_right_lane):
            overlay = draw_lanes(0, overlay, prev_left_lane[:, 0], prev_left_lane[:, 1])
            overlay = draw_lanes(1, overlay, prev_right_lane[:, 0], prev_right_lane[:, 1])
            overlay, center_lane = draw_center_lanes(overlay, prev_left_lane, prev_right_lane)
            both_detected_info = "Previous Lanes: No Detected"
        
        overlay, steer = adjust_vehicle_direction(overlay, center_lane)
        left2heading_distance, right2heading_distance = exception_extract_centerLane_of_bothLanes(overlay)
        current_steer_angle = angle_of_steer(center_lane)
        
        bev_frame = define_world_coordinate(overlay, map_height, map_width)
        both_lane_shape = ipm_lane_info(bev_frame)

        if both_lane_shape == 1:
            text = 'Curve'
        else:
            text = 'Straight'
        
        steer_text = None
        if steer == 1:
            steer_text = 'Straight'
        elif steer == 2:
            steer_text = 'Left'
            current_steer_angle = - current_steer_angle
        elif steer == 3:
            steer_text = 'Right'
        
        if prev_steer_angle is not None and current_steer_angle != prev_steer_angle:
            correct_thres = max(prev_steer_angle, current_steer_angle) - min(prev_steer_angle, current_steer_angle)
            if correct_thres >= 5:
                current_steer_angle = (prev_steer_angle + current_steer_angle) / 2

        steer_angle_pub.publish(np.deg2rad(current_steer_angle)) # steer_angle: '-' if steer_angle < 0  else '+'
        left_heading_distance_pub.publish(left2heading_distance)
        right_heading_distance_pub.publish(right2heading_distance)

        prev_steer_angle = current_steer_angle
        alpha = 1       
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, f'Lane Curvature : {text}' , (0, 100), 1, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Steer : {steer_text}', (0, 125), 1, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Steer Angle : {current_steer_angle}', (0, 145), 1, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Detected Lanes : {both_detected_info}', (0, 165), 1, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Left-Heading Distance: {left2heading_distance}', (0, 180), 1, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Right Heading Distance: {right2heading_distance}', (0, 195), 1, 1, (0, 0, 255), 2)

        # record.write(frame)
        
        cv2.imshow('frame', frame)
        cv2.imshow('BEV frame', bev_frame)
        
        # hardware_check()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    rospy.init_node('lane_detection', anonymous=True)
    
    detect_lines()

if __name__ == "__main__":
    main()
