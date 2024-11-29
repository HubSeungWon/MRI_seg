"""
Qualitative test script of semantic segmentation for OmniDet.

# usage: ./qualitative_semantic.py --config data/params.yaml

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""
import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from PIL import ImageFont, ImageDraw
from main import collect_args
from resnet import ResnetEncoder
from semantic_decoder import SemanticDecoder
from utils import Tupperware
from utils import semantic_color_encoding
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.ndimage import label


FRAME_RATE = 1


def pre_image_op(args, index, frame_index, cam_side):
    total_car1_images = 6054
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))
    if args.crop:
        if int(frame_index[1:]) < total_car1_images:
            cropped_coords = cropped_coords["Car1"][cam_side]
        else:
            cropped_coords = cropped_coords["Car2"][cam_side]
    else:
        cropped_coords = None

    cropped_image = get_image(args, index, cropped_coords, frame_index, cam_side)
    return cropped_image


def get_image(args, index, cropped_coords, frame_index, cam_side):
    recording_folder = "rgb_images" if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)
    image = Image.open(path).convert('RGB')
    if args.crop:
        return image.crop(cropped_coords)
    return image

from PIL import ImageFont
from PIL import ImageDraw

# def load_gt_label(args, image_name):
#     gt_path = os.path.join(args.gtlabel_dir, f"{image_name}.png")
#     gt_image = Image.open(gt_path).convert('L')
#     # gt_image = gt_image.resize((544, 288), Image.LANCZOS)
#     return gt_image

def calculate_level(percentile):
    """오염도를 percentile에 따라 레벨로 변환하는 함수"""
    if percentile < 1:
        return 0
    else:
        return min((percentile // 10) + 1, 10)



# def calculate_iou(predictions, gt_label, class_id):
#     """특정 클래스에 대한 IoU를 계산"""
#     intersection = np.sum((predictions == class_id) & (gt_label == class_id))
#     union = np.sum((predictions == class_id) | (gt_label == class_id))
#     iou = intersection / union if union > 0 else 0
#     return iou



# correct_pixel = [0, 0, 0]



def calculate_center_of_mass(predictions, target_class):
    """특정 클래스의 중심 좌표를 계산"""
    y_coords, x_coords = np.where(predictions == target_class)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None  # 불투명 영역이 없을 경우
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return center_x, center_y


from scipy.ndimage import label

def calculate_center_of_mass_multiple_regions(predictions, target_class):
    """특정 클래스에 속하는 여러 영역의 중심 좌표를 계산"""
    y_coords, x_coords = np.where(predictions == target_class)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return []  # 불투명 영역이 없을 경우

    # 연결된 영역을 식별 (연결된 픽셀 그룹에 라벨 할당)
    labeled_array, num_features = label(predictions == target_class)

    centers = []
    for region in range(1, num_features + 1):
        region_y_coords, region_x_coords = np.where(labeled_array == region)
        if len(region_x_coords) > 0 and len(region_y_coords) > 0:
            center_x = int(np.mean(region_x_coords))
            center_y = int(np.mean(region_y_coords))
            centers.append((center_x, center_y))
    
    return centers




@torch.no_grad()
def test_simple(args):

    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    # encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
    # semantic_decoder_path = os.path.join(args.pretrained_weights, "semantic.pth")

    encoder_path = os.path.join(args.model_path, "encoder.pth")
    semantic_decoder_path = os.path.join(args.model_path, "semantic.pth")

    # print(args.pretrained_weights)
    print(args.model_path)

    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device) # pretrained=False
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)


    # Confusion matrix accumulator (3x3 for 3 classes, adjust size as needed)
    confusion_matrix_accumulator = np.zeros((args.semantic_num_classes, args.semantic_num_classes), dtype=np.int32)


    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("=> Loading pretrained decoder")
    decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(args.device)
    loaded_dict = torch.load(semantic_decoder_path, map_location=args.device)
    decoder.load_state_dict(loaded_dict)
    decoder.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height * 2))

    semantic_color_coding = semantic_color_encoding(args)
    image_paths = [line.rstrip('\n') for line in open(args.val_file)]


    path= "./data/WoodScape_ICCV19/test1004/test_image/*"  # 1003 선정 테스트데이터
    # path = "./data/WoodScape_ICCV19/test_data1002/test_image/*"
    # path = "./data/WoodScape_ICCV19/imagefile_parking_center/*"
    image_paths = glob.glob((path))

    print(image_paths)

    # exit()
    # font_size=16
    font = ImageFont.truetype(font = 'malgun.ttf', size=16)



    # # 각 클래스별 누적 IoU를 저장할 변수 초기화
    # iou_accumulator_class_1 = 0
    # iou_accumulator_class_2 = 0
    # iou_accumulator_class_3 = 0


    # centers_over_frames = []  # 여러 프레임의 중심점을 저장하는 리스트


    semantics = list()
    for idx, image_path in enumerate(image_paths):

        input_image = Image.open(image_path).convert('RGB')
        num_channel = len(input_image.split())
        print("-------------------------------", num_channel)
        print(input_image.size, num_channel)
        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        print(input_image.shape)

        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features)
        semantic_pred = outputs[("semantic", 0)]


        # Segmentation 모델 결과를 Numpy 배열 형식으로 저장하는 코드
        # semantic_pred에서 나온 예측 결과를 처리한 후에, 각 픽셀의 분류된 클래스를 predictions 배열로 변환, 그 후 predictions 배열을 .npy파
        # Saving numpy file
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(args.output_directory, f"{output_name}_semantic.npy")

        # Saving colormapped semantic image
        semantic_pred = semantic_pred[0].data
        print(semantic_pred.shape)
        _, predictions = torch.max(semantic_pred.data, 0)
        predictions = predictions.byte().squeeze().cpu().detach().numpy()
        print("================", predictions.shape)


        # # GT 라벨 불러오기
        # image_name = os.path.splitext(os.path.basename(image_path))[0]
        # gt_label = load_gt_label(args, image_name)
        # gt_label = gt_label.resize((544, 288), Image.LANCZOS)
        # gt_label = np.array(gt_label)


        # PREDICTION
        input_image = input_image.to(args.device)
        features = encoder(input_image)
        outputs = decoder(features)
        semantic_pred = outputs[("semantic", 0)]


        # 0: 클린

        cnt = [0, 0, 0] # 예측값
        # cnt_gt = [0,0,0]
        
        for h in range(288):
            for w in range(544):
                # if predictions[h][w] == 3:
                if predictions[h][w] == 1: # 투명
                    cnt[0] = cnt[0] + 1
                    # if gt_label[h][w] == 1:
                    #     correct_pixel[0] = correct_pixel[0] + 1
                elif predictions[h][w] == 2: # 반투명
                    cnt[1] = cnt[1] + 1
                    # if gt_label[h][w] == 1:
                        # correct_pixel[1] = correct_pixel[1] + 1
                elif predictions[h][w] == 3: # 불투명
                    cnt[2] = cnt[2] + 1
                    # if gt_label[h][w] == 1:
                        # correct_pixel[0] = correct_pixel[0] + 1


                # if gt_label[h][w] == 1:
                #     cnt_gt[0] = cnt_gt[0] + 1
                # elif gt_label[h][w] == 2:
                #     cnt_gt[1] = cnt_gt[1] + 1
                # elif gt_label[h][w] == 3:
                #     cnt_gt[2] = cnt_gt[2] + 1


        # print("correct_pixel :", correct_pixel)

        print(cnt[0], cnt[1], cnt[2],
              (cnt[0]/(544*288)), (cnt[1]/(544*288)), (cnt[2]/(544*288)),
              int((cnt[0]/(544*288)) * 100), int((cnt[1]/(544*288)) * 100), int((cnt[2]/(544*288)) * 100))
        # percentile = int((cnt/(544*288)) * 100)

        percentile = int(((cnt[0] / (544 * 288)) * 0.1) * 100) + int(((cnt[1] / (544 * 288)) * 0.2) * 100) \
        + int(((cnt[2] / (544 * 288)) * 0.7) * 100)

        prob =  (((cnt[0] / (544 * 288)) * 0.1)) + (((cnt[1] / (544 * 288)) * 0.2)) + (((cnt[2] / (544 * 288)) * 0.7))
        print(f'{prob:.6f}')


        # # GT에 대한 percentile 및 오염도 계산
        # percentile_gt = (
        #     int(((cnt_gt[0] / (544 * 288)) * 0.1) * 100) +
        #     int(((cnt_gt[1] / (544 * 288)) * 0.2) * 100) +
        #     int(((cnt_gt[2] / (544 * 288)) * 0.7) * 100)
        # )
        # prob_gt = (
        #     (((cnt_gt[0] / (544 * 288)) * 0.1)) +
        #     (((cnt_gt[1] / (544 * 288)) * 0.2)) +
        #     (((cnt_gt[2] / (544 * 288)) * 0.7))
        # )

        # 예측값과 GT 각각에 대해 오염도 레벨 계산
        level_pred = calculate_level(percentile)
        # level_gt = calculate_level(percentile_gt)

        print("percentile : " ,percentile)
        # print("percentile_gt : ", percentile_gt)





        # 클래스 1, 2, 3에 대한 IoU 계산
        # iou_class_1 = calculate_iou(predictions, gt_label, 1)
        # iou_class_2 = calculate_iou(predictions, gt_label, 2)
        # iou_class_3 = calculate_iou(predictions, gt_label, 3)

        # print(f"Class 1 IoU: {iou_class_1:.4f}")
        # print(f"Class 2 IoU: {iou_class_2:.4f}")
        # print(f"Class 3 IoU: {iou_class_3:.4f}")


        # # IoU 계산 후 누적
        # iou_accumulator_class_1 += iou_class_1
        # iou_accumulator_class_2 += iou_class_2
        # iou_accumulator_class_3 += iou_class_3

        # 현재까지의 클래스별 누적 평균 IoU 출력
        # cumulative_iou_class_1 = iou_accumulator_class_1 / (idx + 1)
        # cumulative_iou_class_2 = iou_accumulator_class_2 / (idx + 1)
        # cumulative_iou_class_3 = iou_accumulator_class_3 / (idx + 1)

        # print(f"Cumulative IoU for Class 1 after {idx + 1} images: {cumulative_iou_class_1:.4f}")
        # print(f"Cumulative IoU for Class 2 after {idx + 1} images: {cumulative_iou_class_2:.4f}")
        # print(f"Cumulative IoU for Class 3 after {idx + 1} images: {cumulative_iou_class_3:.4f}")



        # Flatten the predictions and gt_label to compare pixel-wise
        # flattened_predictions = predictions.flatten()
        # flattened_gt_label = gt_label.flatten()

        # Calculate confusion matrix for the current image
        # current_confusion_matrix = confusion_matrix(flattened_gt_label, flattened_predictions, labels=np.arange(args.semantic_num_classes))

        # Accumulate confusion matrix
        # confusion_matrix_accumulator += current_confusion_matrix

        # 결과 이미지 및 비디오 저장 코드 생략...
        print(f"=> LoL! Beautiful video created and dumped to disk. \n=> Done!")



        print(f"예측 오염도 level={level_pred}")
        # print(f"실제 오염도 level={level_gt}")
        # exit()
        # Save prediction labels
        np.save(name_dest_npy, predictions)
        semantics.append(predictions)

        alpha = 0.5
        color_semantic = np.array(transforms.ToPILImage()(input_image.cpu().squeeze(0)))
        not_background = predictions != 0
        # not_background = predictions == 3
        color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - alpha) +
                                               semantic_color_coding[
                                                   predictions[not_background]] * alpha)
        semantic_color_mapped_pil = Image.fromarray(color_semantic)




        semantic_color_mapped_pil = Image.fromarray(color_semantic)

        # # 4. 중심 좌표 계산 (class 3에 해당하는 영역)
        # center = calculate_center_of_mass(predictions, target_class=3)
        


        # # 6. 중심 좌표가 있을 경우, 동그라미 그리기
        # if center:
        #     center_x, center_y = center
        #     print(f"불투명 오염 영역의 중심점: ({center_x}, {center_y})")
        #     circle_radius = 5
        #     draw = ImageDraw.Draw(semantic_color_mapped_pil)
        #     draw.ellipse((center_x - circle_radius, center_y - circle_radius, 
        #                 center_x + circle_radius, center_y + circle_radius), 
        #                 outline=(255, 0, 0), width=2)
        #     draw.text((center_x + 10, center_y - 10), "Center", fill=(255, 0, 0), font=font)
        # else:
        #     print("클래스 3에 해당하는 픽셀이 없습니다.")



        # 오염도 레벨이 1 이상일 때만 중심점을 계산하고 찍는 코드
        if level_pred >= 1:
            # 4. 각 영역의 중심 좌표 계산 (class 3에 해당하는 영역)
            centers = calculate_center_of_mass_multiple_regions(predictions, target_class=3)
            # 6. 각 중심 좌표에 동그라미 그리기
            if centers:
                # frame_centers = [] # 중심좌표 담을 리스트
                for center_x, center_y in centers:
                    print(f"불투명 오염 영역의 중심점: ({center_x}, {center_y})")
                    circle_radius = 5
                    draw = ImageDraw.Draw(semantic_color_mapped_pil)
                    draw.ellipse((center_x - circle_radius, center_y - circle_radius, 
                                center_x + circle_radius, center_y + circle_radius), 
                                outline=(255, 0, 0), width=2)
                    draw.text((center_x + 10, center_y - 10), "Center", fill=(255, 0, 0), font=font)
                # centers_over_frames.append(frame_centers) # 여러 프레임의 중심점을 모아서 저장
            else:
                print("클래스 3에 해당하는 픽셀이 없습니다.")

        # # 10프레임마다 중심점 리스트를 확인하고 출력
        # if (idx + 1) % 10 == 0:
        #     print(f"Frame {idx + 1}: 10프레임 동안의 중심점 모음:")
        #     for frame_idx, frame_centers in enumerate(centers_over_frames[-10:]):  # 마지막 10프레임의 중심점 출력
        #         print(f"프레임 {frame_idx + idx - 9}: {frame_centers}")

        
        # Colormapped 이미지 저장
        name_dest_im = os.path.join(args.output_directory, f"{output_name}_semantic_level_{level_pred}.png")

        pil_input_image = F.to_pil_image(input_image.cpu().squeeze(0))
        draw = ImageDraw.Draw(semantic_color_mapped_pil)
        draw.text((0, 0), f"오염도 레벨: {level_pred}, {prob:.3f}, 1/{cnt[0]}, 2/{cnt[1]}, 3/{cnt[2]}", (255, 0, 0), font=font)  #RGB
        rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
        rgb_color_pred_concat.paste(pil_input_image, (0, 0))
        rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
        rgb_color_pred_concat.save(name_dest_im)

        rgb_cv2 = np.array(pil_input_image)
        frame = np.concatenate((rgb_cv2, np.array(semantic_color_mapped_pil)), axis=0)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")


    # Print the cumulative confusion matrix
    print("=> Cumulative Confusion Matrix:")
    print(confusion_matrix_accumulator)


    # 최종 누적된 IoU 출력 (모든 이미지 처리 후)
    # print(f"Final Cumulative IoU for Class 1: {cumulative_iou_class_1:.4f}")
    # print(f"Final Cumulative IoU for Class 2: {cumulative_iou_class_2:.4f}")
    # print(f"Final Cumulative IoU for Class 3: {cumulative_iou_class_3:.4f}")


    np.save(os.path.join(args.output_directory, "semantics.npy"), np.concatenate(semantics))

    if args.create_video:
        video.release()

    print(f"=> LoL! Beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)