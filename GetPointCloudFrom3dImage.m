function pts = GetPointCloudFrom3dImage(three_d_image)
    
    cleaned = bwmorph3(imerode(three_d_image,1),'clean');
    cc = bwconncomp(cleaned);
    num_pixels = cellfun(@numel,cc.PixelIdxList);
    [~,I_max] = max(num_pixels);
    ind = cc.PixelIdxList{I_max};
    pts = ind2sub2(size(three_d_image),ind);
    pts = pts - mean(pts);

end