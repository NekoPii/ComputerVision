clc
clear

%% Parameter List [Modify these parameters for tuning operation]
layer_num=5;%Number of Scale Space Layers
sigma=0.8;%ini-sigma
suppress_size=3;%Filter size for local maximization
threshold=0.05;%Threshold used after non-maximum suppression
k=10;%Transformation ratio in scale space

distinction=2;%Enlarge or shrink matrix values as a whole to speed up subsequent operations

img_path="./SIPD/";%Picture save path (not including picture name)
img_name=input("Input the image name : ","s");%Picture name
save_img_type=".bmp";%Save picture type (not recommended to modify)

disp("Running...");

%% Import Image
if ~exist(img_path,"dir")
    mkdir(img_path);
end

ori_img=imread(img_path+img_name);
gray_img=rgb2gray(ori_img);
gray_double_img=im2double(gray_img);%Convert to double precision type to facilitate calculation
[h,w]=size(gray_img);

%% Get Normalized LoG filter
Nor_LoG_filter_size=2*ceil(sigma*3)+1;
%Nor_LoG_filter=fspecial("gaussian",Nor_LoG_filter_size,2*sigma)-fspecial("gaussian",Nor_LoG_filter_size,sigma);
Nor_LoG_filter=(sigma^2)*fspecial("log",Nor_LoG_filter_size,sigma);

%% Calculate Scale Space
scale_space=zeros(h,w,layer_num);
tmp_img=gray_double_img;
for now_layer=1:layer_num
    filtered_img=imfilter(tmp_img,Nor_LoG_filter,"replicate");%Get filtered picture
    filtered_img=filtered_img.^distinction;%Make the result obvious
    scale_space(:,:,now_layer)=imresize(filtered_img,[h,w],"bicubic");%Keep the size of each layer consistent
    tmp_img=imresize(gray_double_img,[ceil(h/(k^now_layer)),ceil(w/(k^now_layer))],"bicubic");%Change the image size of each layer
end

%% Get Local Maximization in [suppress_size*suppress_size*3] Neighbourhood
max_space=zeros(h,w,layer_num);

%each_layer
for now_layer=1:layer_num
    max_space(:,:,now_layer)=ordfilt2(scale_space(:,:,now_layer),suppress_size^2,ones(suppress_size,suppress_size));
end

%every three layers
final_max_space=zeros(h,w,layer_num);
for now_layer=1:layer_num
    for x=1:h
        for y=1:w
            final_max_space(x,y,now_layer)=max(max_space(x,y,[max(1,now_layer-1),min(layer_num,now_layer+1)]));
        end
    end
end

%% Non-maximun Suppression

%Perform non-maximum suppression before thresholding
final_max_space=final_max_space.*(final_max_space==scale_space);
final_max_space=final_max_space.*(final_max_space>threshold);

%% Get [x,y,r] Interest Point
rows=[];
cols=[];
radius=[];
for now_layer=1:layer_num
    [row,col]=find(final_max_space(:,:,now_layer));% row,col in col-order
    now_radius=sigma*(k^(now_layer-1))*sqrt(2);%Get the same radius for each layer
    radius=[radius,repmat(now_radius,[1,size(row)])];
    rows=[rows,row'];
    cols=[cols,col'];
end

%% Draw Circle and Disp
figure(1);
set(gca,'units','normal','pos',[0 0 1 1],'PlotBoxAspectRatioMode','auto','DataAspectRatioMode','auto')
imshow(ori_img);

alpha=0:0.1:2*pi;%draw circle
x=rows+sin(alpha)'.*radius;
y=cols+cos(alpha)'.*radius;
line(y,x,"LineWidth",1.2,"Color","r");

disp("Saving...");
str1=strsplit(img_name,'.');
saveas(gcf,img_path+"Res_"+str1(1)+"_way2_k="+num2str(k)+"_th="+num2str(threshold)+save_img_type);
disp("Completed");
    
disp(int2str(length(radius))+" points of interest detected in the current picture!");
disp("The result image has been saved in "+pwd+"\SIPD\ ");

