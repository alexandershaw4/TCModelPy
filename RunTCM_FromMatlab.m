function [X,BestCost] = RunTCM_FromMatlab(Y0)
% Set up and run TCModel in python, from matlab.
%
% Input Y0 is 4:80 hz amplitude spectrum in 1Hz res (length==77)
%
% Calls the python-based thalamo-cortical model and integration routines
%
% Uses Matlab based optimiser & objective function (here, artificial bee
% colony)
%
% AS

global YY M tcm
YY = Y0;

% tcm [& conda] in MATLAB
% pyversion /Users/Alex/anaconda/bin/python

% PyBP location
modpath = '/Users/Alex/code/TCModelPy';
P       = py.sys.path;
if count(P,modpath) == 0
    insert(P,int32(0),modpath);
end


% Import the TCM tools
tcm = py.importlib.import_module('TCModel');

% Set up a new model
M   = tcm.NewParams();

P  = M{2};
G  = M{3};
M0 = M{1};

% Run it (simulate)
Y  = tcm.RunIntRespond(M0,P,G);
Y  = double(Y{1});

% reversibly convert parameter struct(dict) to array
pE  = tcm.DictToArr(P);
pEv = pE.double;

% fit the model using abc
[X,BestCost] = atcm.optim.abcAS(@Obj,pE.double',(~~pE.double)'/8);


end


function e = Obj(P)
global YY M tcm

P  = py.numpy.array(double(P));         % double  -> np.array
P0 = tcm.ArrToDict(P,M{2});             % np.array->orderedDict (struct)
Y  = tcm.RunIntRespond(M{1},P0{1},M{3});% integration & spectra
e  = spm_vec(Y{1}.double) - spm_vec(YY);% difference
e  = sum(e).^2;                         % squared error

plot(spm_vec(YY),'b','linewidth',2); hold on;
plot(spm_vec(Y{1}.double) ,'r-','linewidth',2); hold off;
drawnow;

end


