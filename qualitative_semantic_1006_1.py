# """
# Qualitative test script of semantic segmentation for OmniDet.

# # usage: ./qualitative_semantic.py --config data/params.yaml

# # author: Varun Ravi Kumar <rvarun7777@gmail.com>

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
# and are not liable for anything.
# """
# import glob
# import os

# import cv2
# import numpy as np
# import torch
# import torchvision.transforms.functional as F
# import yaml
# from PIL import Image
# from torchvision import transforms

# from main import collect_args
# from resnet import ResnetEncoder
# from semantic_decoder import SemanticDecoder
# from utils import Tupperware
# from utils import semantic_color_encoding
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


# FRAME_RATE = 1


# def pre_image_op(args, index, frame_index, cam_side):
#     total_car1_images = 6054
#     cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
#                                     MVL=(343, 5, 1088, 411),
#                                     MVR=(185, 5, 915, 425),
#                                     RV=(186, 203, 1105, 630)),
#                           Car2=dict(FV=(160, 272, 1030, 677),
#                                     MVL=(327, 7, 1096, 410),
#                                     MVR=(175, 4, 935, 404),
#                                     RV=(285, 187, 1000, 572)))
#     if args.crop:
#         if int(frame_index[1:]) < total_car1_images:
#             cropped_coords = cropped_coords["Car1"][cam_side]
#         else:
#             cropped_coords = cropped_coords["Car2"][cam_side]
#     else:
#         cropped_coords = None

#     cropped_image = get_image(args, index, cropped_coords, frame_index, cam_side)
#     return cropped_image


# def get_image(args, index, cropped_coords, frame_index, cam_side):
#     recording_folder = "rgb_images" if index == 0 else "previous_images"
#     file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
#     path = os.path.join(args.dataset_dir, recording_folder, file)
#     image = Image.open(path).convert('RGB')
#     if args.crop:
#         return image.crop(cropped_coords)
#     return image

# from PIL import ImageFont
# from PIL import ImageDraw

# def load_gt_label(args, image_name):
#     gt_path = os.path.join(args.gtlabel_dir, f"{image_name}.png")
#     gt_image = Image.open(gt_path).convert('L')
#     # gt_image = gt_image.resize((544, 288), Image.LANCZOS)
#     return gt_image

# def calculate_level(percentile):
#     """오염도를 percentile에 따라 레벨로 변환하는 함수"""
#     if percentile < 1:
#         return 0
#     elif percentile >= 1 and percentile < 10:
#         return 1
#     elif percentile >= 10 and percentile < 20:
#         return 2
#     elif percentile >= 20 and percentile < 30:
#         return 3
#     elif percentile >= 30 and percentile < 40:
#         return 4
#     elif percentile >= 40 and percentile < 50:
#         return 5
#     elif percentile >= 50 and percentile < 60:
#         return 6
#     elif percentile >= 60 and percentile < 70:
#         return 7
#     elif percentile >= 70 and percentile < 80:
#         return 8
#     elif percentile >= 80 and percentile < 90:
#         return 9
#     elif percentile >= 90:
#         return 10


# # 클래스별 정확도 계산 => 예측과 GT가 일치하는 픽셀 수 / 해당 클래스에 대한 예측된 총 픽셀 수
# def calculate_accuracy_per_class(predictions, gt_label, class_id):
#     """특정 클래스에 대한 정확도를 계산"""
#     # 해당 클래스의 예측과 GT가 일치하는 픽셀 수
#     correct_pixels = np.sum((predictions == class_id) & (gt_label == class_id))
    
#     # 해당 클래스에 대한 예측된 총 픽셀 수
#     total_predicted_pixels = np.sum(predictions == class_id)
    
#     # 해당 클래스의 정확도 계산 (0으로 나누는 경우 처리)
#     accuracy = correct_pixels / total_predicted_pixels if total_predicted_pixels > 0 else 0
#     return accuracy




# # Precision, Recall, F1 Score 계산 함수
# def calculate_precision(tp, fp):
#     return tp / (tp + fp) if (tp + fp) > 0 else 0

# def calculate_recall(tp, fn):
#     return tp / (tp + fn) if (tp + fn) > 0 else 0

# def calculate_f1_score(precision, recall):
#     return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# correct_pixel = [0, 0, 0]




# @torch.no_grad()
# def test_simple(args):


#     """Function to predict for a single image or folder of images"""
#     if not os.path.isdir(args.output_directory):
#         os.mkdir(args.output_directory)

#     # encoder_path = os.path.join(args.pretrained_weights, "encoder.pth")
#     # semantic_decoder_path = os.path.join(args.pretrained_weights, "semantic.pth")

#     encoder_path = os.path.join(args.model_path, "encoder.pth")
#     semantic_decoder_path = os.path.join(args.model_path, "semantic.pth")

#     # print(args.pretrained_weights)
#     print(args.model_path)

#     print("=> Loading pretrained encoder")
#     encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device) # pretrained=False
#     loaded_dict_enc = torch.load(encoder_path, map_location=args.device)

#     # extract the height and width of image that this model was trained with
#     feed_height = loaded_dict_enc['height']
#     feed_width = loaded_dict_enc['width']
#     filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
#     encoder.load_state_dict(filtered_dict_enc)
#     encoder.eval()

#     print("=> Loading pretrained decoder")
#     decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(args.device)
#     loaded_dict = torch.load(semantic_decoder_path, map_location=args.device)
#     decoder.load_state_dict(loaded_dict)
#     decoder.eval()

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
#     video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height * 2))

#     semantic_color_coding = semantic_color_encoding(args)
#     image_paths = [line.rstrip('\n') for line in open(args.val_file)]

#     #path = "./221115_HKMC_CANLAB/100/soil/*"
#     # path = "./data/WoodScape_ICCV19/imagefile_parking2/*"
#     # path = "./data/WoodScape_ICCV19/testdata/*"
#     # path = "./data/WoodScape_ICCV19/test/rgbImages/*"
#     # path = "./data/WoodScape_ICCV19/egg03/*"
#     # path = "./data/WoodScape_ICCV19/test_data1002/test_image/*"
#     # path = "./data/WoodScape_ICCV19/testdata/*"
#     # path = "./data/WoodScape_ICCV19/test_temp/testdata(1차전이_2차전이학습데이터로 테스트)/*"
#     path= "./data/WoodScape_ICCV19/test1004/test_image/*"  # 1003 선정 테스트데이터
#     # path = "./data/WoodScape_ICCV19/test_data1002/test_image/*"
#     image_paths = glob.glob((path))

#     print(image_paths)

#     # exit()
#     # font_size=16
#     font = ImageFont.truetype(font = 'malgun.ttf', size=16)



#     # 각 클래스별 누적 정확도를 저장할 변수 초기화
#     correct_pixel_accumulator_class_1 = 0
#     correct_pixel_accumulator_class_2 = 0
#     correct_pixel_accumulator_class_3 = 0

#     # 각 클래스별 총 픽셀 수를 저장할 변수 초기화
#     total_pixel_accumulator_class_1 = 0
#     total_pixel_accumulator_class_2 = 0
#     total_pixel_accumulator_class_3 = 0



#     # 클래스별 TP, FP, FN을 저장할 변수 초기화
#     tp_accumulator_class_1 = 0
#     fp_accumulator_class_1 = 0
#     fn_accumulator_class_1 = 0

#     tp_accumulator_class_2 = 0
#     fp_accumulator_class_2 = 0
#     fn_accumulator_class_2 = 0

#     tp_accumulator_class_3 = 0
#     fp_accumulator_class_3 = 0
#     fn_accumulator_class_3 = 0





#     semantics = list()
#     for idx, image_path in enumerate(image_paths):




#         # if image_path.endswith(f"_semantic.png"):
#         #     continue
#         # frame_index, cam_side = image_path.split('.')[0].split('_')
#         # input_image = pre_image_op(args, 0, frame_index, cam_side)
#         input_image = Image.open(image_path).convert('RGB')
#         num_channel = len(input_image.split())
#         print("-------------------------------", num_channel)
#         print(input_image.size, num_channel)
#         input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
#         input_image = transforms.ToTensor()(input_image).unsqueeze(0)
#         print(input_image.shape)

#         # PREDICTION
#         input_image = input_image.to(args.device)
#         features = encoder(input_image)
#         outputs = decoder(features)
#         semantic_pred = outputs[("semantic", 0)]


#         # Segmentation 모델 결과를 Numpy 배열 형식으로 저장하는 코드
#         # semantic_pred에서 나온 예측 결과를 처리한 후에, 각 픽셀의 분류된 클래스를 predictions 배열로 변환, 그 후 predictions 배열을 .npy파
#         # Saving numpy file
#         output_name = os.path.splitext(os.path.basename(image_path))[0]
#         name_dest_npy = os.path.join(args.output_directory, f"{output_name}_semantic.npy")

#         # Saving colormapped semantic image
#         semantic_pred = semantic_pred[0].data
#         print(semantic_pred.shape)
#         _, predictions = torch.max(semantic_pred.data, 0)
#         predictions = predictions.byte().squeeze().cpu().detach().numpy()
#         print("================", predictions.shape)


#         # GT 라벨 불러오기
#         image_name = os.path.splitext(os.path.basename(image_path))[0]
#         gt_label = load_gt_label(args, image_name)
#         gt_label = gt_label.resize((544, 288), Image.LANCZOS)
#         gt_label = np.array(gt_label)




#         """
#         prev version: same weights
#         cnt = 0
#         level = 0
#         for h in range(288):
#             for w in range(544):
#                 # if predictions[h][w] == 3:
#                 if predictions[h][w] > 0:
#                     cnt = cnt + 1
#         print(cnt, (cnt/(544*288)), int((cnt/(544*288)) * 100))
#         percentile = int((cnt/(544*288)) * 100)
#         """

#         # 0: 클린

#         cnt = [0, 0, 0] # 예측값
#         cnt_gt = [0,0,0]
        
#         for h in range(288):
#             for w in range(544):
#                 # if predictions[h][w] == 3:
#                 if predictions[h][w] == 1: # 투명
#                     cnt[0] = cnt[0] + 1
#                     if gt_label[h][w] == 1:
#                         correct_pixel[0] = correct_pixel[0] + 1
#                 elif predictions[h][w] == 2: # 반투명
#                     cnt[1] = cnt[1] + 1
#                     if gt_label[h][w] == 1:
#                         correct_pixel[1] = correct_pixel[1] + 1
#                 elif predictions[h][w] == 3: # 불투명
#                     cnt[2] = cnt[2] + 1
#                     if gt_label[h][w] == 1:
#                         correct_pixel[0] = correct_pixel[0] + 1


#                 if gt_label[h][w] == 1:
#                     cnt_gt[0] = cnt_gt[0] + 1
#                 elif gt_label[h][w] == 2:
#                     cnt_gt[1] = cnt_gt[1] + 1
#                 elif gt_label[h][w] == 3:
#                     cnt_gt[2] = cnt_gt[2] + 1


#         print("correct_pixel :", correct_pixel)

#         print(cnt[0], cnt[1], cnt[2],
#               (cnt[0]/(544*288)), (cnt[1]/(544*288)), (cnt[2]/(544*288)),
#               int((cnt[0]/(544*288)) * 100), int((cnt[1]/(544*288)) * 100), int((cnt[2]/(544*288)) * 100))
#         # percentile = int((cnt/(544*288)) * 100)

#         percentile = int(((cnt[0] / (544 * 288)) * 0.1) * 100) + int(((cnt[1] / (544 * 288)) * 0.2) * 100) \
#         + int(((cnt[2] / (544 * 288)) * 0.7) * 100)

#         prob =  (((cnt[0] / (544 * 288)) * 0.1)) + (((cnt[1] / (544 * 288)) * 0.2)) + (((cnt[2] / (544 * 288)) * 0.7))
#         print(f'{prob:.6f}')


#         # GT에 대한 percentile 및 오염도 계산
#         percentile_gt = (
#             int(((cnt_gt[0] / (544 * 288)) * 0.1) * 100) +
#             int(((cnt_gt[1] / (544 * 288)) * 0.2) * 100) +
#             int(((cnt_gt[2] / (544 * 288)) * 0.7) * 100)
#         )
#         prob_gt = (
#             (((cnt_gt[0] / (544 * 288)) * 0.1)) +
#             (((cnt_gt[1] / (544 * 288)) * 0.2)) +
#             (((cnt_gt[2] / (544 * 288)) * 0.7))
#         )

#         # 예측값과 GT 각각에 대해 오염도 레벨 계산
#         level_pred = calculate_level(percentile)
#         level_gt = calculate_level(percentile_gt)

#         print("percentile : " ,percentile)
#         print("percentile_gt : ", percentile_gt)

#         # 클래스 1, 2, 3에 대한 정확도 계산
#         accuracy_class_1 = calculate_accuracy_per_class(predictions, gt_label, 1)
#         accuracy_class_2 = calculate_accuracy_per_class(predictions, gt_label, 2)
#         accuracy_class_3 = calculate_accuracy_per_class(predictions, gt_label, 3)

#         print(f"Class 0 Accuracy: {accuracy_class_1:.4f}")
#         print(f"Class 1 Accuracy: {accuracy_class_2:.4f}")
#         print(f"Class 2 Accuracy: {accuracy_class_3:.4f}")



#         # 각 클래스별 정확도 값을 누적
#         correct_pixel_accumulator_class_1 += accuracy_class_1
#         correct_pixel_accumulator_class_2 += accuracy_class_2
#         correct_pixel_accumulator_class_3 += accuracy_class_3
        
#         total_pixel_accumulator_class_1 += 1  # 각 클래스에 대한 이미지 수 누적
#         total_pixel_accumulator_class_2 += 1
#         total_pixel_accumulator_class_3 += 1

#         # 현재까지의 클래스별 누적 평균 정확도 출력
#         cumulative_accuracy_class_1 = correct_pixel_accumulator_class_1 / total_pixel_accumulator_class_1
#         cumulative_accuracy_class_2 = correct_pixel_accumulator_class_2 / total_pixel_accumulator_class_2
#         cumulative_accuracy_class_3 = correct_pixel_accumulator_class_3 / total_pixel_accumulator_class_3

#         print(f"Cumulative Accuracy for Class 0 after {idx + 1} images: {cumulative_accuracy_class_1:.4f}")
#         print(f"Cumulative Accuracy for Class 1 after {idx + 1} images: {cumulative_accuracy_class_2:.4f}")
#         print(f"Cumulative Accuracy for Class 2 after {idx + 1} images: {cumulative_accuracy_class_3:.4f}")


#         # TP, FP, FN 계산
#         tp_class_1 = np.sum((predictions == 1) & (gt_label == 1))
#         fp_class_1 = np.sum((predictions == 1) & (gt_label != 1))
#         fn_class_1 = np.sum((predictions != 1) & (gt_label == 1))

#         tp_class_2 = np.sum((predictions == 2) & (gt_label == 2))
#         fp_class_2 = np.sum((predictions == 2) & (gt_label != 2))
#         fn_class_2 = np.sum((predictions != 2) & (gt_label == 2))

#         tp_class_3 = np.sum((predictions == 3) & (gt_label == 3))
#         fp_class_3 = np.sum((predictions == 3) & (gt_label != 3))
#         fn_class_3 = np.sum((predictions != 3) & (gt_label == 3))

#         # 누적 TP, FP, FN 업데이트
#         tp_accumulator_class_1 += tp_class_1
#         fp_accumulator_class_1 += fp_class_1
#         fn_accumulator_class_1 += fn_class_1

#         tp_accumulator_class_2 += tp_class_2
#         fp_accumulator_class_2 += fp_class_2
#         fn_accumulator_class_2 += fn_class_2

#         tp_accumulator_class_3 += tp_class_3
#         fp_accumulator_class_3 += fp_class_3
#         fn_accumulator_class_3 += fn_class_3

#         # 현재 이미지에 대한 클래스별 Precision, Recall, F1 score 계산
#         precision_class_1 = calculate_precision(tp_class_1, fp_class_1)
#         recall_class_1 = calculate_recall(tp_class_1, fn_class_1)
#         f1_class_1 = calculate_f1_score(precision_class_1, recall_class_1)

#         precision_class_2 = calculate_precision(tp_class_2, fp_class_2)
#         recall_class_2 = calculate_recall(tp_class_2, fn_class_2)
#         f1_class_2 = calculate_f1_score(precision_class_2, recall_class_2)

#         precision_class_3 = calculate_precision(tp_class_3, fp_class_3)
#         recall_class_3 = calculate_recall(tp_class_3, fn_class_3)
#         f1_class_3 = calculate_f1_score(precision_class_3, recall_class_3)

#         # print(f"Class 1 Precision: {precision_class_1:.4f}, Recall: {recall_class_1:.4f}, F1 Score: {f1_class_1:.4f}")
#         # print(f"Class 2 Precision: {precision_class_2:.4f}, Recall: {recall_class_2:.4f}, F1 Score: {f1_class_2:.4f}")
#         # print(f"Class 3 Precision: {precision_class_3:.4f}, Recall: {recall_class_3:.4f}, F1 Score: {f1_class_3:.4f}")

#         # 누적 Precision, Recall, F1 score 계산
#         cumulative_precision_class_1 = calculate_precision(tp_accumulator_class_1, fp_accumulator_class_1)
#         cumulative_recall_class_1 = calculate_recall(tp_accumulator_class_1, fn_accumulator_class_1)
#         cumulative_f1_class_1 = calculate_f1_score(cumulative_precision_class_1, cumulative_recall_class_1)

#         cumulative_precision_class_2 = calculate_precision(tp_accumulator_class_2, fp_accumulator_class_2)
#         cumulative_recall_class_2 = calculate_recall(tp_accumulator_class_2, fn_accumulator_class_2)
#         cumulative_f1_class_2 = calculate_f1_score(cumulative_precision_class_2, cumulative_recall_class_2)

#         cumulative_precision_class_3 = calculate_precision(tp_accumulator_class_3, fp_accumulator_class_3)
#         cumulative_recall_class_3 = calculate_recall(tp_accumulator_class_3, fn_accumulator_class_3)
#         cumulative_f1_class_3 = calculate_f1_score(cumulative_precision_class_3, cumulative_recall_class_3)

#         print(f"Cumulative Precision for Class 1: {cumulative_precision_class_1:.4f}, "
#             f"Recall: {cumulative_recall_class_1:.4f}, F1 Score: {cumulative_f1_class_1:.4f}")
#         print(f"Cumulative Precision for Class 2: {cumulative_precision_class_2:.4f}, "
#             f"Recall: {cumulative_recall_class_2:.4f}, F1 Score: {cumulative_f1_class_2:.4f}")
#         print(f"Cumulative Precision for Class 3: {cumulative_precision_class_3:.4f}, "
#             f"Recall: {cumulative_recall_class_3:.4f}, F1 Score: {cumulative_f1_class_3:.4f}")





#         # 결과 이미지 및 비디오 저장 코드 생략...
#         print(f"=> LoL! Beautiful video created and dumped to disk. \n=> Done!")





#         print(f"예측 오염도 level={level_pred}")
#         print(f"실제 오염도 level={level_gt}")
#         # exit()
#         # Save prediction labels
#         np.save(name_dest_npy, predictions)
#         semantics.append(predictions)

#         alpha = 0.5
#         color_semantic = np.array(transforms.ToPILImage()(input_image.cpu().squeeze(0)))
#         not_background = predictions != 0
#         # not_background = predictions == 3
#         color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - alpha) +
#                                                semantic_color_coding[
#                                                    predictions[not_background]] * alpha)
#         semantic_color_mapped_pil = Image.fromarray(color_semantic)

#         name_dest_im = os.path.join(args.output_directory, f"{output_name}_semantic_level_{level_pred}.png")

#         pil_input_image = F.to_pil_image(input_image.cpu().squeeze(0))
#         draw = ImageDraw.Draw(semantic_color_mapped_pil)
#         draw.text((0, 0), f"오염도 레벨: {level_pred}, {prob:.3f}, 1/{cnt[0]}, 2/{cnt[1]}, 3/{cnt[2]}", (255, 0, 0), font=font)  #RGB
#         rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
#         rgb_color_pred_concat.paste(pil_input_image, (0, 0))
#         rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
#         rgb_color_pred_concat.save(name_dest_im)

#         rgb_cv2 = np.array(pil_input_image)
#         frame = np.concatenate((rgb_cv2, np.array(semantic_color_mapped_pil)), axis=0)
#         video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#         print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")




#     np.save(os.path.join(args.output_directory, "semantics.npy"), np.concatenate(semantics))

#     if args.create_video:
#         video.release()

#     print(f"=> LoL! Beautiful video created and dumped to disk. \n"
#           f"=> Done!")


# if __name__ == '__main__':
#     config = collect_args()
#     params = yaml.safe_load(open(config.config))
#     args = Tupperware(params)
#     test_simple(args)