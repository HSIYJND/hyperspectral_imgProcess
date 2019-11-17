# 用于进行云检测

import cv2
import numpy as np
import copy
import math


def main():
    land_sea_mask = cv2.imread("D:\\Levir\\804\\dataset\\ground_truth\\landmask.png", 0)

    # 获取海陆分割的掩膜，结果为二值图
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    sea_mask = cv2.erode(land_sea_mask, erode_kernel, 1)
    land_mask = cv2.dilate(land_sea_mask, dilate_kernel, 1)
    # 腐蚀膨胀后不再是二值图，使用阈值分割进行约束
    ret, sea_mask = cv2.threshold(sea_mask, 100, 255, cv2.THRESH_OTSU)  # 海面掩膜
    ret, land_mask = cv2.threshold(land_mask, 100, 255, cv2.THRESH_OTSU)  # 陆地掩膜
    land_mask = 255 - land_mask

    # cv2.namedWindow("land_sea_mask", 0)
    # cv2.imshow("land_sea_mask", land_sea_mask)
    # cv2.namedWindow("sea_mask", 0)
    # cv2.imshow("sea_mask", sea_mask)
    # cv2.namedWindow("land_mask", 0)
    # cv2.imshow("land_mask", land_mask)


    # 海洋云去除波段选取130-139波段
    sea_spectral_index = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139]
    sea_spectral_num = 0
    sea_cloud_list = []
    for i in sea_spectral_index:
        filepath = 'D:\\Levir\\804\\dataset\\804_2100x2048\\selected_png\\' + str(i) + '.png'
        image = cv2.imread(filepath, 0)
        image_sea_with_cloud = cv2.bitwise_and(image, image, mask=sea_mask)  # 分割出海上的区域
        ret, image_sea_cloud = cv2.threshold(image_sea_with_cloud, 100, 255, cv2.THRESH_OTSU)  # 阈值分割出海上的云
        sea_cloud_list.append(image_sea_cloud)

        writepath = 'D:\\Levir\\804\\dataset\\804_2100x2048\\result\\' + 'sea_cloud_' + str(sea_spectral_num) + '.png'
        cv2.imwrite(writepath, image_sea_cloud)
        sea_spectral_num += 1

    # 进行多谱段信息融合
    sea_cloud_result = cv2.bitwise_or(sea_cloud_list[0], sea_cloud_list[1])
    for i in range(len(sea_cloud_list)):
        sea_cloud_result = cv2.bitwise_or(sea_cloud_result, sea_cloud_list[i])

    # 用于显示每个波段的分割结果
    # for i in range(len(sea_cloud_list)):
    #     cv2.namedWindow("sea_cloud", 0)
    #     cv2.imshow("sea_cloud", sea_cloud_list[i])
    #     print(i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    cv2.namedWindow("sea_cloud_result", 0)
    cv2.imshow("sea_cloud_result", sea_cloud_result)
    cv2.imwrite("D:\\Levir\\804\\dataset\\804_2100x2048\\result\\sea_cloud_result.png", sea_cloud_result)


    # 陆地云检测，选择的波段编号间land_cloud_index
    land_spectral_index = [128, 129, 130, 131, 132]
    land_spectral_num = 0
    land_cloud_list = []
    for i in land_spectral_index:
        filepath = 'D:\\Levir\\804\\dataset\\804_2100x2048\\selected_png\\' + str(i) + '.png'
        image = cv2.imread(filepath, 0)
        image_land_with_cloud = cv2.bitwise_and(image, image, mask=land_mask)  # 分割出海上的区域
        ret, image_land_cloud = cv2.threshold(image_land_with_cloud, 100, 255, cv2.THRESH_OTSU)  # 阈值分割出海上的云
        land_cloud_list.append(image_land_cloud)

        writepath = 'D:\\Levir\\804\\dataset\\804_2100x2048\\result\\' + 'land_cloud_' + str(land_spectral_num) + '.png'
        cv2.imwrite(writepath, image_land_cloud)
        land_spectral_num += 1


    # 进行多谱段信息融合
    land_cloud_result = cv2.bitwise_or(land_cloud_list[0], land_cloud_list[1])
    for i in range(len(land_cloud_list)):
        land_cloud_result = cv2.bitwise_or(land_cloud_result, land_cloud_list[i])

    # 用于显示每个波段的分割结果
    # for i in range(len(land_cloud_list)):
    #     cv2.namedWindow("land_cloud", 0)
    #     cv2.imshow("land_cloud", land_cloud_list[i])
    #     print(i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    cv2.namedWindow("land_cloud_result", 0)
    cv2.imshow("land_cloud_result", land_cloud_result)
    cv2.imwrite("D:\\Levir\\804\\dataset\\804_2100x2048\\result\\land_cloud_result.png", land_cloud_result)

    # 海洋陆地云融合
    cloud_detect_result = cv2.bitwise_or(sea_cloud_result, land_cloud_result)
    cv2.namedWindow("cloud_detect_result", 0)
    cv2.imshow("cloud_detect_result", cloud_detect_result)
    cv2.imwrite("D:\\Levir\\804\\dataset\\804_2100x2048\\result\\cloud_detect_result.png", cloud_detect_result)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
