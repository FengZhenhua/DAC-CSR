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

function outShape = projectS2B(inShape, bbox)
inShape = reshape(inShape,[],2);
outShape = inShape;
outShape(:,1) = outShape(:,1) / (max(outShape(:,1)) - min(outShape(:,1))) * bbox(3);
outShape(:,2) = outShape(:,2) / (max(outShape(:,2)) - min(outShape(:,2))) * bbox(4);
outShape(:,1) = outShape(:,1) - min(outShape(:,1)) + bbox(1);
outShape(:,2) = outShape(:,2) - min(outShape(:,2)) + bbox(2);
outShape = outShape(:);
end

