% Demo for DAC-CSR
%
% Copyright @ Zhenhua Feng, fengzhenhua2010@gmail.com
% Centre for Vision, Speech and Signal Processing, University of Surrey
%
% Please cite the following papers if you are using this code
%
% Feng, Z. H., Kittler, J., Christmas, W., Huber, P., & Wu, X. J. (2017, July).
% Dynamic attention-controlled cascaded shape regression exploiting training data augmentation and fuzzy-set sample weighting.
% In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on (pp. 3681-3690). IEEE.
%

close all
clear
clc

mainAFLWDir = './aflw/data/flickr/'; % path to the AFLW dataset

verbose = 1; % set this value to 1 to show fitting examples
run('./vlfeat/toolbox/vl_setup')

% load the AFLW Full protocol data
AFLWinfo = load('./aflw/AFLWinfo_release.mat');

% obtain the test images' information, including their image path, bbox and
% ground truth landmarks for evaluation
testList = AFLWinfo.ra(20001:end);
oriBbox = AFLWinfo.bbox(testList,:);
gtLmk = AFLWinfo.data(testList,:)';
testImg = AFLWinfo.nameList(testList);
for i = 1:length(oriBbox)
    testBbox(:,i) = [oriBbox(i,1); oriBbox(i,3); oriBbox(i,2) - oriBbox(i,1); oriBbox(i,4) - oriBbox(i,3)];
end

% load the pre-trained DAC-CSR model on AFLW
load('./model/daccsr_aflw.mat');

% apply the pre-trained DAC-CSR model to all the test images
preLmk = zeros(size(gtLmk));
timeCost = zeros(1, length(testImg));
for i = 1:length(testImg)
    disp(['Apply DACCSR to the test image id-' num2str(i)]);
    % convert color images to gray level images
    tmpImgO = imread([mainAFLWDir testImg{i}]);
    if size(tmpImgO,3) == 3
        tmpImg = rgb2gray(tmpImgO);
    else
        tmpImg = tmpImgO;
    end
    
    % perform facial landmark detection using the pre-trained model
    tic;
    preLmk(:,i) = fitDaccsr(tmpImg, testBbox(:,i), cr_model);
    timeCost(i) = toc;
    
    % display fitting result
    if verbose
        imshow(tmpImgO);
        hold on;
        bbox = testBbox(:,i);
        plot([bbox(1),bbox(1)+bbox(3),bbox(1)+bbox(3),bbox(1),bbox(1)],[bbox(2),bbox(2),bbox(2)+bbox(4),bbox(2)+bbox(4),bbox(2)],'y-');
        plot(preLmk(1:end/2,i), preLmk(end/2+1:end,i), 'y*');
        plot(gtLmk(1:end/2,i), gtLmk(end/2+1:end,i), 'ro');
        hold off;
        title('Press any key for the next image.');
        pause()
    end
end

% calculate NRMSE
nrmse = [];
for i = 1:length(testImg)
    nrmse(i) = 0;
    for j = 1:size(preLmk,1)/2
        nrmse(i) = nrmse(i) + sqrt((preLmk(j,i) - gtLmk(j,i))^2 + (preLmk(j+end/2,i) - gtLmk(j+end/2,i))^2);
    end
    nrmse(i) = nrmse(i) / testBbox(3,i) / (size(preLmk,1)/2);
end
disp(['Average error AFLW-full: ' num2str(mean(nrmse))]);
disp(['Speed: ' num2str(1/mean(timeCost)) ' fps']);

% plot CED curve
figure();
plot([0 sort(nrmse)], [0, 1:length(nrmse)]/length(nrmse), 'k-', 'linewidth', 2);
xlim([0 0.1]);
ylim([0 1]);
grid on;
box on;
legend('DAC-CSR');
xlabel('Error Normalised by Face Size');
ylabel('Fraction  of Test Faces (4386 in Total)');