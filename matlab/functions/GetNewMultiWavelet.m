function[ng] = GetNewMultiWavelet(g,h,g1,h1)
    ng=[g(:),h(:)];
    [ng]=OtherMakeNewMultiWavelet(ng,g1,h1);  
end

