% 用于转换数据格式，将16bit raw格式图像装换为8bit png格式图像，并调整编号
% 目前804提供的图像数据为0.1-4nm波段，需要做云检测

close all; clc; clear;

read_filepath = '../dataset/804_2100x2048/113-150/';
write_filepath = '../dataset/804_2100x2048/113-150png/';
img_path_list = dir(strcat(read_filepath, '*.raw'));

img_num = length(img_path_list);

for i = 1:img_num
    img_name = strcat(read_filepath, img_path_list(i).name);
    f = fopen(img_name, 'r');
    temp_img = fread(f, 'uint16');  % 读取raw数据
    temp_img2 = reshape(temp_img, 2048, 2100);  % 转换为2100 * 2048
    temp_img2 = temp_img2';
    temp_img2 = temp_img2 / 65535 * 255;  % 转换为uint8数据类型
    temp_img3 = uint8(temp_img2);
    
    index = find(img_path_list(i).name == '.');
    id = str2num(img_path_list(i).name(1:index-1)) + 112;
    write_name = strcat(write_filepath, num2str(id), '.png');
    disp(write_name);
    imwrite(temp_img3, write_name);
end

