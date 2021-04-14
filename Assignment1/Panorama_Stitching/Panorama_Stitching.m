clc
clear

%% VLFeat Environment Build

run("./vlfeat-0.9.21-bin/vlfeat-0.9.21/toolbox/vl_setup");
disp("VLFeat Version: "+num2str(vl_version));

%% Import Imgs and Size Unification
img_path="./PanoStich/";%Picture save path (not including picture name)
img1_name=input("Input the First Image's Name:","s");%Picture1 name
img2_name=input("Input the Second Image's Name:","s");
save_img_type=".bmp";%Save picture type (not recommended to modify)

disp("Running...");

if ~exist(img_path,"dir")
    mkdir(img_path);
end

img1=imread(img_path+img1_name);
img2=imread(img_path+img2_name);


[h1,w1]=size(img1,1:2);
[h2,w2]=size(img2,1:2);

%Find The MaxH and MaxW To Resize Two Images
h=max(h1,h2);
w=max(w1,w2);

img1=imresize(img1,[h,w]);
img2=imresize(img2,[h,w]);

gray_img1=rgb2gray(img1);
gray_img2=rgb2gray(img2);

single_gray_img1=im2single(gray_img1);
single_gray_img2=im2single(gray_img2);

%% VLFeat---SIFT [To Find the Key Points]

%[f,d]=vl_sift(single_gray_img) 
%Where f is a 4*n vector matrix ,every column f=[x,y,s,th], 
%The center position of the point of interest is (x,y)
%The scale is s
%The gradient direction is th
%Where d represents a 128-dimensional feature vector

[f1,descr1]=vl_sift(single_gray_img1);
[f2,descr2]=vl_sift(single_gray_img2);

Interest_Points_nums=200;

%Get the random scrambled column number arrangement of f
col_rand_1=randperm(size(f1,2));
col_rand_2=randperm(size(f2,2));

%Select the last Interest_Points_nums points of random scrambled column
backSection_col_rand_1=col_rand_1(end-Interest_Points_nums:end);
backSection_col_rand_2=col_rand_2(end-Interest_Points_nums:end);

%Get the last Interest_Points_nums points of interest randomly shuffled in column order
rand_points1=f1(:,backSection_col_rand_1);
rand_points2=f2(:,backSection_col_rand_2);

%% Show Two Images' Interest Points with VLFeat——SIFT
figure();
set(gcf,"position",[0 0 1920 1080]);

%Remove the extra white edges of the Figure [See tight_subplot.m for details]
hs=tight_subplot(2,2,[0.1, 0.05], 0.1, 0.01);

axes(hs(1));
imshow(img1,[]);
set(vl_plotframe(rand_points1),"color","b","lineWidth",1.2);
title("First Img Interest Points");
axis("off");

axes(hs(2));
imshow(img2,[]);
set(vl_plotframe(rand_points2),"color","b","lineWidth",1.2);
title("Second Img Interest Points");
axis("off");

%% Match Two Img Interest Points by descr

%[match,dis] = vl_ubcmatch(descr1, descr2) 
% For each descriptor in the matrix descr1, vl_ubcmatch finds the closest one to it in descr2 (use Euclidean distance to judge)
% Retuns the matches and also the squared Euclidean distance between the matches.
% match is a 2*n matrix, the value of each column is the serial number of descr1 and descr2
% dis is the distance between descr1 and descr2

[match,dis]=vl_ubcmatch(descr1,descr2);

% dis=sorted_dis(dis_index) [Sort inverse operation]
[sorted_dis,dis_index]=sort(dis,"descend");

% Correspond the sorted dis with matches
match=match(:,dis_index);

% Find the point coordinates in the order of matches
match1_points=f1(1:2,match(1,:));
match2_points=f2(1:2,match(2,:));

%% RANSAC with HomographyEstimation
%initialize
cnt=200;% Lookup round
threshold=5;
match_num=size(match,2);% Number of matches
max_inlier_count=0;% Maximum number of inliers
now_inlier_cnt=0;% Current number of inliers
H_final=zeros(3,3);% The final H determined after the iteration

for i=1:cnt
    %Select 4 random points by Random sample
    sample_index=randsample(size(match,2),4);% Get the Sample index
    fir_points=match1_points(:,sample_index)'; % Get Img1(First image) 4 random match points
    sec_points=match2_points(:,sample_index)'; % Get Img2(Second image) 4 random match points
    
    %   Img1      <--->    Img2
    %fir1[fx1,fy1]<--->sec1[sx1,sy1]
    %fir2[fx2,fy2]<--->sec2[sx2,sy2]
    %fir3[fx3,fy3]<--->sec3[sx3,sy3]
    %fir4[fx4,fy4]<--->sec4[sx4,sy4]
    
    % firn=[xn,yn] secn=[xn,yn]
    fir1=fir_points(1,:);
    fir2=fir_points(2,:);
    fir3=fir_points(3,:);
    fir4=fir_points(4,:);
    sec1=sec_points(1,:);
    sec2=sec_points(2,:);
    sec3=sec_points(3,:);
    sec4=sec_points(4,:);
    
    % Homography
    X=[
        sec1(1) sec1(2) 1 0 0 0 -sec1(1)*fir1(1) -sec1(2)*fir1(1) -fir1(1);
        0 0 0 sec1(1) sec1(2) 1 -sec1(1)*fir1(2) -sec1(2)*fir1(2) -fir1(2);
        sec2(1) sec2(2) 1 0 0 0 -sec2(1)*fir2(1) -sec2(2)*fir2(1) -fir2(1);
        0 0 0 sec2(1) sec2(2) 1 -sec2(1)*fir2(2) -sec2(2)*fir2(2) -fir2(2);
        sec3(1) sec3(2) 1 0 0 0 -sec3(1)*fir3(1) -sec3(2)*fir3(1) -fir3(1);
        0 0 0 sec3(1) sec3(2) 1 -sec3(1)*fir3(2) -sec3(2)*fir3(2) -fir3(2);
        sec4(1) sec4(2) 1 0 0 0 -sec4(1)*fir4(1) -sec4(2)*fir4(1) -fir4(1);
        0 0 0 sec4(1) sec4(2) 1 -sec4(1)*fir4(2) -sec4(2)*fir4(2) -fir4(2);
        ];
    
    % [V,D] = eig(A) 
    % returns the diagonal matrix D and matrix V of eigenvalues
    % Where D is the diagonal matrix of eigenvalues (the eigenvalues are in descending order along the main diagonal)
    % V is the eigen matrix composed of eigenvectors (column vectors) corresponding to the eigenvalues of D
    % The least square solution is V(1)
    
    [V,D]=eig(X'*X);
    H=reshape(V(:,1),3,3)';% Convert the feature vector (column vector) to a 3 by 3 matrix
    H=H/H(end);% make H(3,3)=1 To get the ratio better
    
    % Assess the current H —— [LSM(Least squares method)]
    val1=[match1_points;ones(1,match_num)];
    val2=H*[match2_points;ones(1,match_num)];
    val2=val2./val2(end,:);% make val2(end,:)=1 to make the third dimension no effect
    eps=val2-val1;
    eps=sqrt(sum(eps.^2,1));
    
    % Get the H_Best abd the inline pos index
    isInline=eps<threshold;
    now_inlier_cnt=size(find(isInline),2);
    if now_inlier_cnt>max_inlier_count
        max_inlier_count=now_inlier_cnt;
        inline_pos=find(isInline);
        H_final=H;
    end
end

%% Show Two Images' Matches

true_match1_points=match1_points(:,inline_pos);
true_match2_points=match2_points(:,inline_pos);

axes(hs(3));
% Place img1 and img2 side by side in the same image
showMatchedFeatures(img1,img2,true_match1_points',true_match2_points',"montage");
title("Key Points Matches  [Total "+num2str(max_inlier_count)+" Matches!]");
legend("Left Match Points","Right Match Points");
axis("off");

%% Stiching
% Get the geometric transformation structure of H_final'
% Although the value of tform is the same as H_final'
tform=projective2d(H_final');

% Find the maximum and minimum values of the output space limit
[xlim(1,:), ylim(1,:)] = outputLimits(tform, [1 w], [1 h]);
xMin = min([1; xlim(:)]);
xMax = max([w; xlim(:)]);
yMin = min([1; ylim(:)]);
yMax = max([h; ylim(:)]);

% The width and height of the panorama
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Generate a panorama of empty data
res_img = zeros([height width 3], 'like', img1);

blender = vision.AlphaBlender('Operation', 'Binary mask','MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Add the original image of Img1 to the panorama
img1_trans = imwarp(img1,projective2d(eye(3)),"OutputView", panoramaView);
res_img = step(blender, res_img, img1_trans, img1_trans(:,:,1));

% Perform H_final' geometric transformation on img2
img2_trans = imwarp(img2, tform,"OutputView", panoramaView);
res_img = step(blender, res_img, img2_trans, img2_trans(:,:,1));


axes(hs(4));
imshow(res_img,[]);
title("Panorama Stitching Image");
axis("off");

disp("Done!")

str1=strsplit(img1_name,'.');
str2=strsplit(img2_name,'.');

saveas(gcf,img_path+"Res_"+str1(1)+"_"+str2(1)+save_img_type);

disp("The result image has been saved in "+pwd+"\PanoStich\ ");


