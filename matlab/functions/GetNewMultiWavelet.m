function[ng] = GetNewMultiWavelet(g,h,g1,h1)
    ng=[g(:),h(:)];
   % ng=[g(:);h(:)];       % wc
    [ng]=OtherMakeNewMultiWavelet(ng,g1,h1);       % 16
%     [ng]=OtherMakeNewMultiWavelet(ng,g1,h1);      % 16*4
% %     [ng]=OtherMakeNewMultiWavelet(ng,g1,h1);     % 16*4*4
% %     [ng]=OtherMakeNewMultiWavelet(ng,g1,h1);      % 16*4*4*4
%     G=ng; 
%     S=0;
%     for k=1:size(G,2)
%          S=S+conv(G(:,k),wrev(G(:,k)));
%     end
end

