% Demo for DAC-CSR - project the mean shape to a face bbox
%
% Copyright @ Zhenhua Feng, fengzhenhua2010@gmail.com
% Centre for Vision, Speech and Signal Processing, University of Surrey
%
% Please cite the following papers if you are using this code
%
% Feng, Z. H., Kittler, J., Christmas, W., Huber, P., & Wu, X. J. (2017, July).
% Dynamic attention-controlled cascaded shape regression exploiting training data augmentation and fuzzy-set sample weighting.
% In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on (pp. 3681-3690). IEEE.

function preLmk = fitDaccsr(img, initBbox, csrModel)

% bbox refinement
yList = round(initBbox(2):initBbox(2)+initBbox(4));
xList = round(initBbox(1):initBbox(1)+initBbox(3));
yList(yList < 1) = 1;
xList(xList < 1) = 1;
yList(yList > size(img,1)) = size(img,1);
xList(xList > size(img,2)) = size(img,2);
tmpImg = img(yList, xList);

tmpImg = imresize(tmpImg,[100 100]);
tmp_f1 = vl_hog(single(tmpImg), 10);
tmp_f2 = vl_lbp(single(tmpImg), 10);
feature  = [tmp_f1(:); tmp_f2(:)];

bboxUpdate = (csrModel.model(1).A_bb(1:end-1,:)' * feature + csrModel.model(1).A_bb(end,:)');

initBbox = [bboxUpdate(1)*initBbox(3) + initBbox(1); bboxUpdate(2)*initBbox(4) + initBbox(2); bboxUpdate(3)*initBbox(3); bboxUpdate(4)*initBbox(4)];
preLmk = projectS2B(csrModel.mean_shape, initBbox);

% apply the general CSR models for landmark update
for i = csrModel.sdt_list(1).list
    feature = obtainFeatures(img, preLmk, csrModel.patch_radius);
    ancher = max((max(preLmk(1:end/2)) - min(preLmk(1:end/2))), (max(preLmk(end/2+1:end)) - min(preLmk(end/2+1:end))));
    preLmk = preLmk + ancher * (csrModel.model(i).A(1:end-1,:)' * feature + csrModel.model(i).A(end,:)');
end

% PCA for model selection
tmp_pre_shape = preLmk;
tmp_pre_shape(1:end/2) = tmp_pre_shape(1:end/2) - min(tmp_pre_shape(1:end/2));
tmp_pre_shape(1:end/2) = tmp_pre_shape(1:end/2) / (max(tmp_pre_shape(1:end/2)) - min(tmp_pre_shape(1:end/2)));
tmp_pre_shape(end/2+1:end) = tmp_pre_shape(end/2+1:end) - min(tmp_pre_shape(end/2+1:end));
tmp_pre_shape(end/2+1:end) = tmp_pre_shape(end/2+1:end) / (max(tmp_pre_shape(end/2+1:end)) - min(tmp_pre_shape(end/2+1:end)));

tmp_pre_coef = csrModel.model(1).split_pca_coef \ (tmp_pre_shape - csrModel.model(1).mean_aligned_shape);

mean_coe = csrModel.model(1).mean_aligned_coef;
std_coe = csrModel.model(1).std_aligned_coef;

if tmp_pre_coef(1) >= mean_coe(1) - std_coe(1) && tmp_pre_coef(1) <= mean_coe(1) + std_coe(1) && tmp_pre_coef(2) <= mean_coe(2) + std_coe(2) && tmp_pre_coef(2) >= mean_coe(2) - std_coe(2)
    flag_branch = 5;
elseif tmp_pre_coef(1) <= mean_coe(1) && tmp_pre_coef(2) <= mean_coe(2)
    flag_branch = 1;
elseif tmp_pre_coef(1) <= mean_coe(1) && tmp_pre_coef(2) >= mean_coe(2)
    flag_branch = 2;
elseif tmp_pre_coef(1) >= mean_coe(1)  && tmp_pre_coef(2) >= mean_coe(2)
    flag_branch = 3;
else
    flag_branch = 4;
end

% apply domain specific CSR models for landmark update and online domain
% selection
for i = csrModel.sdt_list(2).list
    feature = obtainFeatures(img, preLmk, csrModel.patch_radius);
    
    ancher = max((max(preLmk(1:end/2)) - min(preLmk(1:end/2))), (max(preLmk(end/2+1:end)) - min(preLmk(end/2+1:end))));
    
    preLmk = preLmk + ancher * (csrModel.model(i).A(flag_branch).data(1:end-1,:)' * feature + csrModel.model(i).A(flag_branch).data(end,:)');
    
    tmp_pre_shape = preLmk;
    tmp_pre_shape(1:end/2) = tmp_pre_shape(1:end/2) - min(tmp_pre_shape(1:end/2));
    tmp_pre_shape(1:end/2) = tmp_pre_shape(1:end/2) / (max(tmp_pre_shape(1:end/2)) - min(tmp_pre_shape(1:end/2)));
    tmp_pre_shape(end/2+1:end) = tmp_pre_shape(end/2+1:end) - min(tmp_pre_shape(end/2+1:end));
    tmp_pre_shape(end/2+1:end) = tmp_pre_shape(end/2+1:end) / (max(tmp_pre_shape(end/2+1:end)) - min(tmp_pre_shape(end/2+1:end)));
    
    tmp_pre_coef = csrModel.model(1).split_pca_coef \ (tmp_pre_shape - csrModel.model(1).mean_aligned_shape);
    
    if tmp_pre_coef(1) >= mean_coe(1) - std_coe(1) && tmp_pre_coef(1) <= mean_coe(1) + std_coe(1) && tmp_pre_coef(2) <= mean_coe(2) + std_coe(2) && tmp_pre_coef(2) >= mean_coe(2) - std_coe(2)
        flag_branch = 5;
    elseif tmp_pre_coef(1) <= mean_coe(1) && tmp_pre_coef(2) <= mean_coe(2)
        flag_branch = 1;
    elseif tmp_pre_coef(1) <= mean_coe(1) && tmp_pre_coef(2) >= mean_coe(2)
        flag_branch = 2;
    elseif tmp_pre_coef(1) >= mean_coe(1)  && tmp_pre_coef(2) >= mean_coe(2)
        flag_branch = 3;
    else
        flag_branch = 4;
    end
    
end

preLmk = preLmk(:);
end

% subfunction for feature extraction
function features = obtainFeatures(img, shape, patch_radius)
patch_radius = max(max(shape(1:end/2))-min(shape(1:end/2)), max(shape(end/2+1:end))-min(shape(end/2+1:end)))/patch_radius;
patch_radius = round(patch_radius);

features = [];
minx = round(min(shape(1:end/2))) - patch_radius;
maxx = round(max(shape(1:end/2))) - patch_radius;
miny = round(min(shape(end/2+1:end))) + patch_radius;
maxy = round(max(shape(end/2+1:end))) + patch_radius;

if minx < 1
    minx = 1;
end
if miny < 1
    miny = 1;
end
if maxx > size(img,2)
    maxx = size(img,2);
end
if maxy > size(img,1)
    maxy = size(img,1);
end
tmp = img(miny:maxy, minx:maxx);
tmp = imresize(tmp, [100 100]);
hogf = vl_hog(single(tmp), 10);
features = [features; hogf(:)];
hogf = vl_lbp(single(tmp), 10);
features = [features; hogf(:)];

for i=1:size(shape,1)/2
    x = floor(shape(i));
    y = floor(shape(i+end/2));
    IX_X = x - patch_radius + 1 : x + patch_radius;
    IX_Y = y - patch_radius + 1 : y + patch_radius;
    IX_Y(IX_Y<1) = 1;
    IX_X(IX_X<1) = 1;
    IX_Y(IX_Y>size(img,1)) = size(img,1);
    IX_X(IX_X>size(img,2)) = size(img,2);
    tmp = img(round(IX_Y), round(IX_X),:);
    tmp = imresize(tmp, [32 32]);
    hogf = vl_hog(single(tmp), 10);
    features = [features; hogf(:)];
    tmp = tmp(8:22, 8:22);
    hogf = vl_hog(single(tmp), 5);
    features = [features; hogf(:)];
end
end