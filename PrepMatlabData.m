% prepare data from an existing DCM for use in my python framework
y = DCM.xY.y{1}; y = y(:);
w = DCM.xY.Hz;   w = w(:);
D.w = w;
D.y = y;
save('~/Desktop/ModelData','D');