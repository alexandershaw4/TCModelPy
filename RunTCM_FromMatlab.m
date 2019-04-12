function [X,BestCost] = RunTCM_FromMatlab(Y0,P0)
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

% TCM py code location
modpath = '/Users/Alex/code/TCModelPy';
P       = py.sys.path;
if count(P,modpath) == 0
    insert(P,int32(0),modpath);
end

% Import the TCM tools
tcm = py.importlib.import_module('TC6');

% Set up a new model
M   = tcm.DefaultParams();

% unpack these (everything returned in 1 var from python)
P  = M{2};
G  = M{3};
M0 = M{1};


% Run it (simulate)
Y  = tcm.RunIntRespond(M0,P,G);
Y  = double(Y{1});

% Package P (neural) & G (observer) params
pE  = tcm.DictToArr(P);


% % Re-arrange input mlab structure to same order as py OrderedDict
% Flds = P.keys;
% MStr = fieldnames(P0);
% for i = 1:length(Flds)
%     NewP.(Flds{i}.string) = P0.( Flds{i}.string() );
% end
% pE = py.numpy.array(double(spm_vec(NewP))); 

% 
[Qp,Cp,Eh,F] = fitmodel(@Run,pE.double',Y0,1:length(Y0))

% % fit the model using abc
% [X,BestCost] = atcm.optim.abcAS(@Obj,pE.double',(~~pE.double)'/8);
% 
% % fit the model using abc
% [X,BestCost] = atcm.optim.abcAS(@Obj,pE.double',ones(size(pE.double))/8);


end


function e = Obj(P)
global YY M tcm 


P  = py.numpy.array(double(P));         % double  -> np.array
P0 = tcm.ArrToDict(P,M{2});             % np.array->orderedDict (cf struct)
Y  = tcm.RunIntRespond(M{1},P0{1},M{3});% integration & spectra
e  = spm_vec(Y{1}.double) - spm_vec(YY);% difference
e  = sum(e).^2;                         % squared error

plot(spm_vec(YY),'b','linewidth',2); hold on;
plot(spm_vec(Y{1}.double) ,'r-','linewidth',2); hold off;
drawnow;

end

function y = Run(P,varargin)
global M tcm 


P  = py.numpy.array(double(P));         % double  -> np.array
P0 = tcm.ArrToDict(P,M{2});             % np.array->orderedDict (cf struct)
y  = tcm.RunIntRespond(M{1},P0{1},M{3});% integration & spectra
y  = y{1}.double();

end

function [Qp,Cp,Eh,F] = fitmodel(fun,x0,Y0,w)


% Set it up as per a dynamic model
M    = [];
M.IS  = fun;
M.pE = x0;
M.pC = diag(ones(size(x0)))/8;

xU    = [];
xU.u  = 0;

xY.y  = {Y0};
xY.Hz = w;
M.x   = [];
M.m   = 1;
M.l   = 1;

% Call the external Bayesian EM routine
[Qp,Cp,Eh,F] = spm_nlsi_GN(M,xU,xY);
end

